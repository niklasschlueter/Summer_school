#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import math
import argparse
import threading
import pinocchio as pin
import time

from utils import print_out_collision_results, get_ee_position_and_rotation, update_pinocchio, MovingAverageFilter
from test_il_policy_v2 import DeploymentPolicy
from geometry_msgs.msg import WrenchStamped
from other.rosbag_recorder import RosbagRecorder
from utils import StateLogger

from il_trainer import ILTrainer
from model import ILPolicy

class CubicSplineTrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('ur_trajectory_planner')
        
        # Publisher for position commands
        self.position_publisher = self.create_publisher(
            Float64MultiArray,
            '/forward_position_controller/commands',
            10
        )
        
        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Subscribe to the wrench topic
        self.ft_subscriber = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.wrench_callback,
            10  # QoS depth
        )
        
        self.current_positions = None
        self.current_velocities = None
        self.current_efforts = None
        self.filtered_current_efforts = np.zeros(6)

        self.efforts_filter = MovingAverageFilter(window_size=100, vector_size=6)

        self.zero_ee_force = np.zeros(3)
        self.zero_ee_torque = np.zeros(3)

        self.ee_force = np.zeros(3)
        self.ee_torque = np.zeros(3)

        self.filtered_ee_force = np.zeros(3)
        self.filtered_ee_torque = np.zeros(3)

        self.forces_filter = MovingAverageFilter(window_size=250, vector_size=3)
        self.torques_filter = MovingAverageFilter(window_size=250, vector_size=3)


        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Control parameters
        self.trajectory_duration = 5.0  # seconds
        self.control_frequency = 500.0   # Hz
        self.dt = 1.0 / self.control_frequency
        
        # Trajectory execution variables
        self.executing_trajectory = False
        self.trajectory_timer = None
        self.trajectory_points = []
        self.current_trajectory_index = 0


        # Pinocchio setup
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf("ur5e.urdf")
        self.data, self.collision_data, self.visual_data = pin.createDatas(
            self.model, self.collision_model, self.visual_model
        )

        # setup_collision_model
        self.collision_model.addAllCollisionPairs()

        pin.removeCollisionPairs(self.model, self.collision_model, "ur5e.srdf")
        print(
            "num collision pairs - after removing useless collision pairs:",
            len(self.collision_model.collisionPairs),
        )

        print("num collision pairs - initial:", len(self.collision_model.collisionPairs))

        # not sure if necessary
        self.collision_data = pin.GeometryData(self.collision_model)

        self.policy = DeploymentPolicy()

        self.state_logger = StateLogger()


        self.beta = 0.9  # Exploration rate for DAgger


    def joint_state_callback(self, msg: JointState):
        """Efficiently update joint state (position, velocity, effort)"""
        # Build a map from joint name to index (only once)
        if not hasattr(self, "_joint_name_to_index"):
            self._joint_name_to_index = {name: i for i, name in enumerate(msg.name)}

        # Preallocate arrays
        n = len(self.joint_names)
        positions = np.zeros(n)
        velocities = np.zeros(n)
        efforts   = np.zeros(n)

        for i, joint_name in enumerate(self.joint_names):
            idx = self._joint_name_to_index.get(joint_name, None)
            if idx is not None:
                if idx < len(msg.position):
                    positions[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    velocities[i] = msg.velocity[idx]
                if idx < len(msg.effort):
                    efforts[i] = msg.effort[idx]

        self.current_positions = positions
        self.current_velocities = velocities
        self.current_efforts = efforts
        self.filtered_current_efforts = self.efforts_filter.update(efforts)

    def set_current_ft_measuerements_as_zero(self):
        self.zero_ee_force = self.ee_force# - self.zero_ee_force
        self.zero_ee_torque = self.ee_torque# - self.zero_ee_torque



    def wrench_callback(self, msg: WrenchStamped):
        """Callback for force/torque sensor data"""
        force = msg.wrench.force
        torque = msg.wrench.torque

        self.ee_force = np.array([force.x, force.y, force.z])
        self.ee_torque = np.array([torque.x, torque.y, torque.z])

        self.ee_force = self.ee_force - self.zero_ee_force
        self.ee_torque = self.ee_torque - self.zero_ee_torque


        self.filtered_ee_force = self.forces_filter.update(self.ee_force)
        self.filtered_ee_torque = self.torques_filter.update(self.ee_torque)

        #self.get_logger().info(
        #print(f"Force: {self.ee_force}, Torque: {self.ee_torque}")
        #)
    
    def cubic_spline_coefficients(self, q0, q1, qd0=0.0, qd1=0.0, duration=1.0):
        """Calculate cubic spline coefficients for position trajectory"""
        T = duration
        
        a0 = q0
        a1 = qd0
        a2 = (3*(q1 - q0)/T**2) - (2*qd0/T) - (qd1/T)
        a3 = (2*(q0 - q1)/T**3) + (qd0/T**2) + (qd1/T**2)
        
        return [a0, a1, a2, a3]
    
    def evaluate_cubic_spline(self, coeffs, t):
        """Evaluate cubic spline at time t"""
        a0, a1, a2, a3 = coeffs
        return a0 + a1*t + a2*t**2 + a3*t**3
    
    def plan_trajectory(self, start_positions, end_positions, duration=None):
        """Plan cubic spline trajectory from start to end positions"""
        if duration is None:
            duration = self.trajectory_duration
            
        # Calculate cubic spline coefficients for each joint
        joint_coefficients = []
        for i in range(6):
            coeffs = self.cubic_spline_coefficients(
                start_positions[i], 
                end_positions[i], 
                duration=duration
            )
            joint_coefficients.append(coeffs)
        
        # Generate trajectory points
        trajectory_points = []
        num_points = int(duration * self.control_frequency)
        
        for step in range(num_points + 1):
            t = step * self.dt
            positions = []
            
            for joint_coeffs in joint_coefficients:
                pos = self.evaluate_cubic_spline(joint_coeffs, t)
                positions.append(pos)
            
            trajectory_points.append((t, positions))
        
        return trajectory_points
    
    def execute_trajectory_point(self):
        """Execute one point of the trajectory"""
        #print(f"current position: {self.current_positions}")
        collision_flag = self.check_for_collisions(self.current_positions)

        if collision_flag or (self.current_trajectory_index >= len(self.trajectory_points) or 
            not self.executing_trajectory):
            self.stop_trajectory()
            return
        
        _time, positions = self.trajectory_points[self.current_trajectory_index]
        
        msg = Float64MultiArray()
        msg.data = positions
        self.position_publisher.publish(msg)
        
        if self.current_trajectory_index % 500 == 0:  # Log every 0.5 seconds
            self.get_logger().info(f"Executing trajectory: {self.current_trajectory_index + 1}"
                                  f"/{len(self.trajectory_points)} points, t={_time:.2f}s")
        
        self.current_trajectory_index += 1

        self.state_logger.log(
            t=time.perf_counter(),
            pos=self.current_positions,
            vel=self.current_velocities,
            eff=self.filtered_current_efforts,
            ft=np.concatenate((self.filtered_ee_force, self.filtered_ee_torque)),
            action=positions - self.current_positions# Delta action used!
        )

    def compute_trajectory_point(self):
        """Execute one point of the trajectory"""
        #print(f"current position: {self.current_positions}")
        collision_flag = self.check_for_collisions(self.current_positions)

        if collision_flag or (self.current_trajectory_index >= len(self.trajectory_points) or 
            not self.executing_trajectory):
            self.stop_trajectory()
            return
        
        _time, positions = self.trajectory_points[self.current_trajectory_index]
        return positions
        
    def publish_action(self, positions):
        msg = Float64MultiArray()
        msg.data = positions
        self.position_publisher.publish(msg)

    def compute_policy(self):

        """Execute one point of the trajectory"""
        #print(f"current position: {self.current_positions}")
        collision_flag = self.check_for_collisions(self.current_positions)

        if collision_flag or not self.executing_trajectory:
            self.stop_trajectory()
            return

        ##print(f"current trajectory index: {self.current_trajectory_index}, "
        ##      f"trajectory length: {len(self.trajectory_points)}")

        #if collision_flag or (self.current_trajectory_index >= len(self.trajectory_points) or 
        #    not self.executing_trajectory):
        #    self.stop_trajectory()
        #    return
        

        x = np.concatenate((self.current_positions, self.current_velocities, self.filtered_current_efforts, self.filtered_ee_force, self.filtered_ee_torque)).astype(np.float32)
        #print(f"np.shape(x): {np.shape(x)}")
        #x = np.concatenate((self.current_positions, self.current_velocities)).astype(np.float32)#, np.zeros_like(self.filtered_current_efforts), np.zeros(3), np.zeros(3))).astype(np.float32)
        #print(f"x: {x}")
        pred = self.policy.run(x)
        #print(f"Policy inference time: {t1 - t0:.4f} seconds")
        positions = self.current_positions + pred


        return positions


    def run_policy(self):

        """Execute one point of the trajectory"""
        #print(f"current position: {self.current_positions}")
        collision_flag = self.check_for_collisions(self.current_positions)

        if collision_flag or not self.executing_trajectory:
            self.stop_trajectory()
            return

        ##print(f"current trajectory index: {self.current_trajectory_index}, "
        ##      f"trajectory length: {len(self.trajectory_points)}")

        #if collision_flag or (self.current_trajectory_index >= len(self.trajectory_points) or 
        #    not self.executing_trajectory):
        #    self.stop_trajectory()
        #    return
        

        x = np.concatenate((self.current_positions, self.current_velocities, self.filtered_current_efforts, self.filtered_ee_force, self.filtered_ee_torque)).astype(np.float32)
        #print(f"np.shape(x): {np.shape(x)}")
        #x = np.concatenate((self.current_positions, self.current_velocities)).astype(np.float32)#, np.zeros_like(self.filtered_current_efforts), np.zeros(3), np.zeros(3))).astype(np.float32)
        #print(f"x: {x}")
        pred = self.policy.run(x)
        #print(f"Policy inference time: {t1 - t0:.4f} seconds")
        positions = self.current_positions + pred


        #positions = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])

        #print(f"output positions: {positions}")
        
        
        msg = Float64MultiArray()
        msg.data = positions
        self.position_publisher.publish(msg)
        
        #if self.current_trajectory_index % 25 == 0:  # Log every 0.5 seconds
        #    self.get_logger().info(f"Executing trajectory: {self.current_trajectory_index + 1}"
        #                          f"/{len(self.trajectory_points)} points, t={time:.2f}s")
        
        self.current_trajectory_index += 1

        self.state_logger.log(
            t=time.perf_counter(),
            pos=self.current_positions,
            vel=self.current_velocities,
            eff=self.filtered_current_efforts,
            ft=np.concatenate((self.filtered_ee_force, self.filtered_ee_torque)),
            action=pred # Delta action used!
        )
    
    def start_trajectory(self, target_positions, duration=None):
        """Start executing a trajectory to target positions"""
        if self.current_positions is None:
            self.get_logger().error("Current positions not available!")
            return False
        
        if self.executing_trajectory:
            self.stop_trajectory()
        
        self.get_logger().info("Planning trajectory...")
        
        self.trajectory_points = self.plan_trajectory(
            self.current_positions, 
            target_positions, 
            duration
        )
        
        self.get_logger().info(f"Planned trajectory: {len(self.trajectory_points)} points, "
                              f"{duration or self.trajectory_duration:.2f}s duration")
        
        self.executing_trajectory = True
        self.current_trajectory_index = 0


        #self.state_logger.reset()
        
        self.trajectory_timer = self.create_timer(self.dt, self.execute_trajectory_point)
        
        return True

    def start_policy(self):
        """Start executing a trajectory to target positions"""
        if self.current_positions is None:
            self.get_logger().error("Current positions not available!")
            return False
        
        if self.executing_trajectory:
            self.stop_trajectory()
        
        self.executing_trajectory = True
        self.current_trajectory_index = 0

        #self.state_logger.reset()
        
        self.trajectory_timer = self.create_timer(self.dt, self.run_policy)
        
        return True

    def start_dagger(self, target_positions, duration=None):
        """Start executing a trajectory to target positions"""
        if self.current_positions is None:
            self.get_logger().error("Current positions not available!")
            return False
        
        if self.executing_trajectory:
            self.stop_trajectory()
        
        self.get_logger().info("Planning trajectory...")
        
        self.trajectory_points = self.plan_trajectory(
            self.current_positions, 
            target_positions, 
            duration
        )
        
        self.get_logger().info(f"Planned trajectory: {len(self.trajectory_points)} points, "
                              f"{duration or self.trajectory_duration:.2f}s duration")
        
        self.executing_trajectory = True
        self.current_trajectory_index = 0
        
        self.trajectory_timer = self.create_timer(self.dt, self.run_dagger)
        
        return True

    def run_dagger(self):
        collision_flag = self.check_for_collisions(self.current_positions)

        if collision_flag or (self.current_trajectory_index >= len(self.trajectory_points) or 
            not self.executing_trajectory):
            self.stop_trajectory()
            return

        action_expert = self.compute_trajectory_point()
        action_policy = self.compute_policy()

        if np.random.uniform(0,1) < self.beta:
            self.publish_action(action_expert)
        else:
            self.publish_action(action_policy)

        #if self.current_trajectory_index % 500 == 0:  # Log every 0.5 seconds
        #    self.get_logger().info(f"Executing trajectory: {self.current_trajectory_index + 1}"
        #                          f"/{len(self.trajectory_points)} points, t={_time:.2f}s")
        
        self.current_trajectory_index += 1

        self.state_logger.log(
            t=time.perf_counter(),
            pos=self.current_positions,
            vel=self.current_velocities,
            eff=self.filtered_current_efforts,
            ft=np.concatenate((self.filtered_ee_force, self.filtered_ee_torque)),
            action=action_expert - self.current_positions# Delta action used!
        )
    
    def stop_trajectory(self):
        """Stop trajectory execution"""
        self.executing_trajectory = False
        if self.trajectory_timer:
            self.trajectory_timer.cancel()
            self.trajectory_timer = None
        self.get_logger().info("Trajectory execution completed")
        # Save data
        #self.state_logger.save("trajectory_data")
        
    def is_trajectory_finished(self):
        """Check if trajectory execution is finished"""
        return not self.executing_trajectory
    
    def wait_for_trajectory_completion(self):
        """Block until trajectory is finished"""
        while not self.is_trajectory_finished():
            time.sleep(0.1)

    def check_for_collisions(self, q):
        self.model, self.data, self.collision_model, self.collision_data, self.visual_model, self.visual_data = update_pinocchio(self.model, self.data, self.collision_model, self.collision_data, self.visual_model, self.visual_data, q)

        collision_flag = pin.computeCollisions(self.model, self.data, self.collision_model, self.collision_data, q, False)
        if collision_flag:
            return True

        #print(f"\nCollision check results: {collision_flag}")
        #print_out_collision_results(self.collision_model, self.collision_data)

        position, rotation = get_ee_position_and_rotation(self.model, self.data, q, update=False)
        if position[2] < 0.10:
            print(f"collision detected: end-effector too low at z={position[2]:.3f} m")
            collision_flag = True

        return collision_flag



def print_positions(positions, label="Positions"):
    """Print positions in a nice format"""
    if positions is None:
        print(f"{label}: Not available")
        return
    
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    print(f"{label}:")
    for name, pos in zip(joint_names, positions):
        print(f"  {name}: {pos:7.3f} rad ({math.degrees(pos):6.1f}°)")


def main():
    # Predefined positions
    predefined_positions = {
        'home': [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
        'ready': [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        'up': [0.0, -0.5, -1.0, -1.5, 0.0, 0.0],
        'side': [1.57, -1.57, 0.0, -1.57, 0.0, 0.0],
        #'crash': [0.0, -1.57, 3.14, -1.57, 0.0, 0.0],
        #"darrens_home": [0.0, -0.734, 1.45, -2.38, -1.76, 0.817],
        "darrens_home": [0.0, -0.934, 1.45, -2.38, -1.76, 0.817],
        "position_1": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        "position_2": [1.57, -1.57, 1.57, -1.57, -1.57, 0.0],
    }

    duration = 2.0
    #target_positions = predefined_positions['darrens_home']  # Default to 'ready' position
    #target_positions = predefined_positions['home']  # Default to 'ready' position
    
    #print(f"Target positions: {[f'{p:.3f}' for p in target_positions]}")
    
    # Initialize ROS2
    rclpy.init()
    
    planner = CubicSplineTrajectoryPlanner()
    
    # Start ROS2 spinning in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(planner,))
    spin_thread.daemon = True
    spin_thread.start()
    
    print("\nWaiting for current joint positions...")
    
    # Wait for joint states
    while planner.current_positions is None:
        time.sleep(0.1)
    
    print("Current positions received!")
    print_positions(planner.current_positions, "Current position")
    #print_positions(target_positions, "Target position")


    def run_expert(data_dir="runs_7", episodes=1, record=True):
            planner.set_current_ft_measuerements_as_zero()
            for episode in range(episodes):
                # Delete all the old data
                planner.state_logger.reset() 

                print(f"\n EPISODE {episode}")

                start_position = predefined_positions["darrens_home"] + np.random.uniform(-0.0, 0.0, size=6)
                if planner.start_trajectory(start_position, duration):
                    t0 = time.perf_counter()
                    #print("Trajectory started! Waiting for completion...")
                    print(f"FORWARD")

                    # Wait for trajectory to finish
                    planner.wait_for_trajectory_completion()

                    print("\n✅ Trajectory completed successfully!")
                    print_positions(planner.current_positions, "Final position")
                    t1 = time.perf_counter()
                    print(f"Total execution time: {t1 - t0:.2f} seconds")
                    planner.stop_trajectory()
                    #recorder.stop()

                else:
                    print("❌ Failed to start trajectory!")
                    return 1

                end_position = predefined_positions["position_2"] + np.random.uniform(-0.0, 0.0, size=6)
                if planner.start_trajectory(end_position, duration):
                    t0 = time.perf_counter()
                    print(f"BACKWARD")

                    # Wait for trajectory to finish
                    planner.wait_for_trajectory_completion()

                    print("\n✅ Trajectory completed successfully!")
                    print_positions(planner.current_positions, "Final position")
                    t1 = time.perf_counter()
                    print(f"Total execution time: {t1 - t0:.2f} seconds")
                    planner.stop_trajectory()
                    #recorder.stop()

                else:
                    print("❌ Failed to start trajectory!")
                    return 1



            if record:
                file_path = f"runs/{data_dir}/run_{episode}.npz" 
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                planner.state_logger.save(file_path)


    def run_policy(data_dir="runs_7", episodes=1, move_to_start_position=False, record=True):
        planner.set_current_ft_measuerements_as_zero()

        for episode in range(episodes):
            if move_to_start_position:
                end_position = predefined_positions["position_2"] + np.random.uniform(-0.0, 0.0, size=6)
                if planner.start_trajectory(end_position, duration):
                    t0 = time.perf_counter()
                    print(f"BACKWARD")

                    # Wait for trajectory to finish
                    planner.wait_for_trajectory_completion()

                    print("\n✅ Trajectory completed successfully!")
                    print_positions(planner.current_positions, "Final position")
                    t1 = time.perf_counter()
                    print(f"Total execution time: {t1 - t0:.2f} seconds")
                    planner.stop_trajectory()
                    #recorder.stop()

                else:
                    print("❌ Failed to start trajectory!")
                    return 1

            ##############################
            #### RUN POLICY
            ##############################

            # Delete all the old data
            planner.state_logger.reset() 

            print(f"\n Policy EPISODE {episode}")

            if planner.start_policy():
                t0 = time.perf_counter()
                print("Trajectory started! Waiting for completion...")

                # Wait for trajectory to finish
                planner.wait_for_trajectory_completion()

                print("\n✅ Trajectory completed successfully!")
                print_positions(planner.current_positions, "Final position")
                t1 = time.perf_counter()
                print(f"Total execution time: {t1 - t0:.2f} seconds")
                planner.stop_trajectory()

            else:
                print("❌ Failed to start trajectory!")
                return 1


            if record: 
                file_path = f"runs/{data_dir}/run_policy_{episode}.npz" 
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                planner.state_logger.save(file_path)

    def run_dagger(data_dir="runs_7", episodes=10, record=True, move_to_start_position=True, start_idx=0, episodes_offset=0):
            planner.set_current_ft_measuerements_as_zero()
            for episode in range(episodes_offset, episodes):
                # Delete all the old data

                if move_to_start_position:
                    end_position = predefined_positions["position_2"] + np.random.uniform(-0.0, 0.0, size=6)
                    if planner.start_trajectory(end_position, duration):
                        t0 = time.perf_counter()
                        print(f"BACKWARD")

                        # Wait for trajectory to finish
                        planner.wait_for_trajectory_completion()

                        print("\n✅ Trajectory completed successfully!")
                        print_positions(planner.current_positions, "Final position")
                        t1 = time.perf_counter()
                        print(f"Total execution time: {t1 - t0:.2f} seconds")
                        planner.stop_trajectory()

                    else:
                        print("❌ Failed to start trajectory!")
                        return 1

                planner.state_logger.reset() 

                print(f"\n EPISODE {episode}")

                start_position = predefined_positions["darrens_home"] + np.random.normal(0, 0.01, size=6)
                if planner.start_dagger(start_position, duration):
                    t0 = time.perf_counter()
                    #print("Trajectory started! Waiting for completion...")
                    print(f"FORWARD")

                    # Wait for trajectory to finish
                    planner.wait_for_trajectory_completion()

                    print("\n✅ Trajectory completed successfully!")
                    print_positions(planner.current_positions, "Final position")
                    t1 = time.perf_counter()
                    print(f"Total execution time: {t1 - t0:.2f} seconds")
                    planner.stop_trajectory()
                    #recorder.stop()

                else:
                    print("❌ Failed to start trajectory!")
                    return 1

                end_position = predefined_positions["position_2"] + np.random.normal(0.0, 0.01, size=6)
                if planner.start_dagger(end_position, duration):
                    t0 = time.perf_counter()
                    print(f"BACKWARD")

                    # Wait for trajectory to finish
                    planner.wait_for_trajectory_completion()

                    print("\n✅ Trajectory completed successfully!")
                    print_positions(planner.current_positions, "Final position")
                    t1 = time.perf_counter()
                    print(f"Total execution time: {t1 - t0:.2f} seconds")
                    planner.stop_trajectory()
                    #recorder.stop()

                else:
                    print("❌ Failed to start trajectory!")
                    return 1



                if record:
                    file_path = f"runs/{data_dir}/run_{start_idx+episode}.npz" 
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    planner.state_logger.save(file_path)

    try:
        trainer = ILTrainer(
            out_dir="models/dagger_model",
            model_ctor=ILPolicy,      # or ILPolicyV2
            model_kwargs={},          # extra args for your model class if needed
            normalize=True,
            seed=0,
            amp=False,                # set True if training on CUDA and you want mixed precision
            lr=1e-3,
            weight_decay=1e-4,
        )
        iterations = 5
        episodes = 50 #50
        for iter in range(1, iterations):
            planner.beta = max(0.1, 1.0 - iter*0.3)
            run_dagger(data_dir="runs_dagger", episodes=episodes, start_idx=iter*episodes, episodes_offset=30)

            trainer.fit(
                data_path="runs/runs_dagger",
                pattern="run_*.npz",
                hidden=64,
                layers=4,
                dropout=0.1,
                batch_size=512,
                epochs=50,                # run however many you want now
                val_ratio=0.2,
                num_workers=4,
                noise_std=0.005,
                grad_clip=1.0,
                control_mode="position_delta",
                min_episodes=2,
                seed=0,
            )
            planner.policy = DeploymentPolicy(path=f"models/dagger_model/best.pt")


            
    except KeyboardInterrupt:
        print(f"Trajectory interrupted by user")
        planner.stop_trajectory()
    
    except Exception as e:
        print(f"Error during trajectory execution: {e}")
        planner.stop_trajectory()
        return 1
    
    finally:
        print("Shutting down...")
        planner.destroy_node()
        rclpy.shutdown()
        spin_thread.join()  # Ensure spinning thread stops before exit
    
    return 0

if __name__ == '__main__':


    main()