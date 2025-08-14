#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import math
import argparse
import time
import threading

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
        
        self.current_positions = None
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        # Control parameters
        self.trajectory_duration = 5.0  # seconds
        self.control_frequency = 250.0   # Hz
        self.dt = 1.0 / self.control_frequency
        
        # Trajectory execution variables
        self.executing_trajectory = False
        self.trajectory_timer = None
        self.trajectory_points = []
        self.current_trajectory_index = 0
        
    def joint_state_callback(self, msg):
        """Update current joint positions from joint_states topic"""
        if len(msg.position) >= 6:
            # Find the indices of our joints in the joint_states message
            joint_positions = [0.0] * 6
            for i, joint_name in enumerate(self.joint_names):
                try:
                    joint_idx = msg.name.index(joint_name)
                    joint_positions[i] = msg.position[joint_idx]
                except ValueError:
                    # Joint not found, use the position by index if available
                    if i < len(msg.position):
                        joint_positions[i] = msg.position[i]
            
            self.current_positions = joint_positions
    
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
        if (self.current_trajectory_index >= len(self.trajectory_points) or 
            not self.executing_trajectory):
            self.stop_trajectory()
            return
        
        time, positions = self.trajectory_points[self.current_trajectory_index]
        
        msg = Float64MultiArray()
        msg.data = positions
        self.position_publisher.publish(msg)
        
        if self.current_trajectory_index % 25 == 0:  # Log every 0.5 seconds
            self.get_logger().info(f"Executing trajectory: {self.current_trajectory_index + 1}"
                                  f"/{len(self.trajectory_points)} points, t={time:.2f}s")
        
        self.current_trajectory_index += 1
    
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
        
        self.trajectory_timer = self.create_timer(self.dt, self.execute_trajectory_point)
        
        return True
    
    def stop_trajectory(self):
        """Stop trajectory execution"""
        self.executing_trajectory = False
        if self.trajectory_timer:
            self.trajectory_timer.cancel()
            self.trajectory_timer = None
        self.get_logger().info("Trajectory execution completed")
    
    def is_trajectory_finished(self):
        """Check if trajectory execution is finished"""
        return not self.executing_trajectory
    
    def wait_for_trajectory_completion(self):
        """Block until trajectory is finished"""
        while not self.is_trajectory_finished():
            time.sleep(0.1)

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
    }

    duration = 3.0
    target_positions = predefined_positions['up']  # Default to 'ready' position
    
    print(f"Target positions: {[f'{p:.3f}' for p in target_positions]}")
    
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
    print_positions(target_positions, "Target position")
    
    try:
        print(f"\nStarting trajectory execution...")
        
        if planner.start_trajectory(target_positions, duration):
            t0 = time.perf_counter()
            print("Trajectory started! Waiting for completion...")
            
            # Wait for trajectory to finish
            planner.wait_for_trajectory_completion()
            
            print("\n✅ Trajectory completed successfully!")
            print_positions(planner.current_positions, "Final position")
            t1 = time.perf_counter()
            print(f"Total execution time: {t1 - t0:.2f} seconds")
            
        else:
            print("❌ Failed to start trajectory!")
            return 1
            
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