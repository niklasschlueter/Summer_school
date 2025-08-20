import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.serialization import serialize_message
from std_msgs.msg import String
import rosbag2_py
from sensor_msgs.msg import JointState, Image, LaserScan, Imu
from geometry_msgs.msg import Twist, PoseStamped, Wrench, WrenchStamped
from nav_msgs.msg import Odometry
import importlib

class MultiBagRecorder(Node):
    def __init__(self, topics_config, bag_name='my_bag'):
        """
        Initialize the multi-topic bag recorder.
        
        Args:
            topics_config (list): List of dictionaries with topic configurations
                Each dict should have: 'name', 'type', 'msg_class'
            bag_name (str): Name of the output bag file
        """
        super().__init__('multi_bag_recorder')
        
        # Initialize bag writer
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=bag_name,
            storage_id='mcap')
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)
        
        # Store subscriptions to prevent garbage collection
        self.subs= []
        
        # Create topics and subscriptions
        for i, topic_config in enumerate(topics_config):
            # Create topic metadata
            topic_info = rosbag2_py.TopicMetadata(
                id=i,
                name=topic_config['name'],
                type=topic_config['type'],
                serialization_format='cdr')
            self.writer.create_topic(topic_info)
            
            # Create subscription
            subscription = self.create_subscription(
                topic_config['msg_class'],
                topic_config['name'],
                lambda msg, topic_name=topic_config['name']: self.topic_callback(msg, topic_name),
                10)
            self.subs.append(subscription)
            
            self.get_logger().info(f"Recording topic: {topic_config['name']}")

    def topic_callback(self, msg, topic_name):
        """Callback for recording messages from any topic."""
        self.writer.write(
            topic_name,
            serialize_message(msg),
            self.get_clock().now().nanoseconds)

def get_message_class(type_string):
    """
    Dynamically import message class from type string.
    
    Args:
        type_string (str): Message type like 'sensor_msgs/msg/JointState'
    
    Returns:
        Message class
    """
    # Common message type mappings
    message_map = {
        'sensor_msgs/msg/JointState': JointState,
        'sensor_msgs/msg/Image': Image,
        'sensor_msgs/msg/LaserScan': LaserScan,
        'sensor_msgs/msg/Imu': Imu,
        'geometry_msgs/msg/Twist': Twist,
        'geometry_msgs/msg/PoseStamped': PoseStamped,
        'nav_msgs/msg/Odometry': Odometry,
        'std_msgs/msg/String': String,
        'geometry_msgs/msg/Wrench': Wrench,
        'geometry_msgs/msg/WrenchStamped': WrenchStamped,
    }
    
    if type_string in message_map:
        return message_map[type_string]
    
    # Try dynamic import for other message types
    try:
        parts = type_string.split('/')
        package_name = parts[0]
        msg_name = parts[2]
        module = importlib.import_module(f'{package_name}.msg')
        return getattr(module, msg_name)
    except (ImportError, AttributeError, IndexError):
        raise ValueError(f"Could not import message type: {type_string}")

def create_topic_config(topic_name, msg_type):
    """
    Helper function to create topic configuration.
    
    Args:
        topic_name (str): ROS topic name (e.g., '/joint_states')
        msg_type (str): Message type (e.g., 'sensor_msgs/msg/JointState')
    
    Returns:
        dict: Topic configuration dictionary
    """
    return {
        'name': topic_name,
        'type': msg_type,
        'msg_class': get_message_class(msg_type)
    }

def main(args=None):
    """
    Main function with example usage.
    Modify the topics_to_record list to record different topics.
    """
    try:
        rclpy.init()
        
        # Configure topics to record
        # Simply modify this list to record different topics
        topics_to_record = [
            create_topic_config('/joint_states', 'sensor_msgs/msg/JointState'),
            create_topic_config('/force_torque_sensor_broadcaster/wrench', 'geometry_msgs/msg/WrenchStamped'),
            #create_topic_config('/cmd_vel', 'geometry_msgs/msg/Twist'),
            #create_topic_config('/odom', 'nav_msgs/msg/Odometry'),
            # Add more topics as needed:
            # create_topic_config('/scan', 'sensor_msgs/msg/LaserScan'),
            # create_topic_config('/camera/image_raw', 'sensor_msgs/msg/Image'),
            # create_topic_config('/imu', 'sensor_msgs/msg/Imu'),
        ]
        
        # Optional: specify custom bag name
        bag_name = 'bags/test'
        
        recorder = MultiBagRecorder(topics_to_record, bag_name)
        rclpy.spin(recorder)
        
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()