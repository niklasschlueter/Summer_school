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
import multiprocessing
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

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
    def __init__(self, joint_states_csv='joint_states_log.csv', speed_scaling_csv='speed_scaling_log.csv'):
        super().__init__('ur_data_recorder')

        self.joint_states_csv = joint_states_csv
        self.speed_scaling_csv = speed_scaling_csv

        qos_profile = QoSProfile(depth=1000, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.joint_states_data = []
        self.speed_scaling_data = []
        self._cached_joint_names = None
        self.lock = threading.Lock()

        self.create_subscription(JointState, '/joint_states', self.joint_states_callback, qos_profile)
        self.create_subscription(Float64, '/speed_scaling_state_broadcaster/speed_scaling', self.speed_scaling_callback, qos_profile)

    def joint_states_callback(self, msg: JointState):
        if self._cached_joint_names is None:
            self._cached_joint_names = tuple(msg.name)
        ts = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            self.joint_states_data.append((ts, msg.position, msg.velocity, msg.effort))

    def speed_scaling_callback(self, msg: Float64):
        ts = self.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            self.speed_scaling_data.append((ts, msg.data))

    def save_data(self):
        print(f"saving data")
        with self.lock:
            joint_states_df = pd.DataFrame(self.joint_states_data, columns=['timestamp', 'position', 'velocity', 'effort'])
            if self._cached_joint_names:
                joint_states_df.attrs['joint_names'] = self._cached_joint_names
            speed_scaling_df = pd.DataFrame(self.speed_scaling_data, columns=['timestamp', 'speed_scaling'])

        joint_states_df.to_csv(self.joint_states_csv, index=False)
        speed_scaling_df.to_csv(self.speed_scaling_csv, index=False)
        self.get_logger().info(f"Data saved to {self.joint_states_csv} and {self.speed_scaling_csv}")

    def destroy_node(self):
        # Save before shutting down
        self.save_data()
        super().destroy_node()


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

def run_logger():
    rclpy.init()
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Exception in logger: {e}")
    node.destroy_node()
    rclpy.shutdown()

def run_robot():
    rclpy.init()
    # Node for sending trajectories
    node = URTrajectoryExecutor()
    #robot_executor = rclpy.executors.SingleThreadedExecutor()
    #robot_executor.add_node(robot)
    time.sleep(1)  # Give time for the node to initialize



    t = JointTrajectory()
    t.joint_names = ROBOT_JOINTS
    t.points = [
        JointTrajectoryPoint(
            positions=[pos-np.pi/4 for pos in HOME],
            velocities=[0.0] * 6,
            time_from_start=Duration(sec=5, nanosec=0)
        ),
        JointTrajectoryPoint(
            positions=[pos+np.pi/4 for pos in HOME],
            velocities=[0.0] * 6,
            time_from_start=Duration(sec=10, nanosec=0)
        )
    ]

    # Start executor in a separate thread
    #exec_thread = threading.Thread(target=executor.spin, daemon=True)
    #exec_thread.start()

    # Give subscribers time to warm up
    #time.sleep(1)

    # Send trajectory
    node.send_trajectory(t)

    ## Stop executor
    #executor.shutdown()
    #exec_thread.join(timeout=1)

    node.destroy_node()
    #rclpy.shutdown()
    # Stop executors
    #recorder_executor.shutdown()
    #robot_executor.shutdown()
    #recorder_thread.join(timeout=1)

    # Save data
    #joint_states_df = recorder.get_joint_states_df()
    #speed_scaling_df = recorder.get_speed_scaling_df()
    #joint_states_df.to_csv('joint_states_log.csv', index=False)
    #speed_scaling_df.to_csv('speed_scaling_log.csv', index=False)
    #robot.get_feedback_df().to_csv('trajectory_feedback_log.csv', index=False)
    #print('Data saved to joint_states_log.csv and speed_scaling_log.csv')

    rclpy.shutdown()


def main():
    multiprocessing.set_start_method('spawn')  # must be at the very top
    logger_proc = multiprocessing.Process(target=run_logger)
    robot_proc = multiprocessing.Process(target=run_robot)

    logger_proc.start()
    robot_proc.start()

    robot_proc.join()
    logger_proc.terminate()


if __name__ == '__main__':
    main()