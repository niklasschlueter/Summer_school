import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import pandas as pd
import numpy as np
import threading
import time

from pathlib import Path
project_dir = Path(__file__).resolve().parents[0]
data_dir = project_dir / "data"
plots_dir = project_dir / "plots"

ROBOT_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

HOME = [0, -np.pi/2, 0, -np.pi/2, 0, 0]

class DataRecorder(Node):
    def __init__(self):
        super().__init__('ur_data_recorder')
        self.joint_states_data = []
        self.speed_scaling_data = []
        self.lock = threading.Lock()
        self.sub_joint_states = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.sub_speed_scaling = self.create_subscription(
            Float64,
            '/speed_scaling_state_broadcaster/speed_scaling',
            self.speed_scaling_callback,
            10
        )
    
    # Both topics are published at 500 Hz, but rclpy is not necessarily able to keep up with that rate.
    def joint_states_callback(self, msg: JointState):
        with self.lock:
            self.joint_states_data.append({
                'timestamp': self.get_clock().now().nanoseconds * 1e-9,
                'name': list(msg.name),
                'position': list(msg.position),
                'velocity': list(msg.velocity),
                'effort': list(msg.effort)
            })

    def speed_scaling_callback(self, msg: Float64):
        with self.lock:
            self.speed_scaling_data.append({
                'timestamp': self.get_clock().now().nanoseconds * 1e-9,
                'speed_scaling': msg.data
            })

    def get_joint_states_df(self):
        with self.lock:
            return pd.DataFrame(self.joint_states_data)

    def get_speed_scaling_df(self):
        with self.lock:
            return pd.DataFrame(self.speed_scaling_data)

class URTrajectoryExecutor(Node):
    def __init__(self):
        super().__init__('ur_trajectory_executor')
        # Can be changed to whatever motion controller you want. Just be aware of what action definition the controller uses.
        # passthrough_trajectory_controller uses the same action definition, and might lighten the load on your computer, but does not publish feedback.
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.trajectory_client.wait_for_server()
        self.feedback = []

    def send_trajectory(self, trajectory: JointTrajectory):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
        future = self.trajectory_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Trajectory goal rejected')
            return None
        print('Trajectory goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        print('Trajectory execution finished')
        return result_future.result()
    
    # Feedback is published at 20 Hz. The topic subsciption above is faster, but not very consistent, in terms of timing.
    def feedback_callback(self, feedback):
        feedback_msg = feedback.feedback
        self.feedback.append({
            'timestamp': self.get_clock().now().nanoseconds * 1e-9,
            'joint_names': feedback_msg.joint_names,
            'desired_positions': list(feedback_msg.desired.positions),
            'actual_positions': list(feedback_msg.actual.positions),
            'error_positions': list(feedback_msg.error.positions),
            'desired_velocities': list(feedback_msg.desired.velocities),
            'actual_velocities': list(feedback_msg.actual.velocities),
        })

    def get_feedback_df(self):
        return pd.DataFrame(self.feedback)

def main():
    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor()
    recorder = DataRecorder()
    robot = URTrajectoryExecutor()
    executor.add_node(recorder)
    executor.add_node(robot)

    t = JointTrajectory()
    t.joint_names = ROBOT_JOINTS
    t.points = [
        JointTrajectoryPoint(
            positions=[pos-np.pi/4 for pos in HOME],
            #positions=[0, 0, 0, 0, 0, 0],
            velocities=[0.0] * 6,
            time_from_start=Duration(sec=5, nanosec=0)
        ),
        JointTrajectoryPoint(
            positions=[pos+np.pi/4 for pos in HOME],
            #positions=[0.5, 0.4, 1.3, 1.0, 1.1, 2.0],
            velocities=[0.0] * 6,
            time_from_start=Duration(sec=10, nanosec=0)
        )
    ]

    # Start executor in a separate thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    # Give subscribers time to warm up
    time.sleep(1)

    # Send trajectory
    robot.send_trajectory(t)

    # Stop executor
    executor.shutdown()
    exec_thread.join(timeout=1)

    # Save data
    joint_states_df = recorder.get_joint_states_df()
    speed_scaling_df = recorder.get_speed_scaling_df()
    joint_states_df.to_csv(data_dir / 'joint_states_log.csv', index=False)
    speed_scaling_df.to_csv(data_dir / 'speed_scaling_log.csv', index=False)
    robot.get_feedback_df().to_csv(data_dir / 'trajectory_feedback_log.csv', index=False)
    print('Data saved to joint_states_log.csv and speed_scaling_log.csv')

    rclpy.shutdown()

if __name__ == '__main__':
    main()