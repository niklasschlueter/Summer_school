import sys
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message

import rosbag2_py


class SimpleBagRecorder(Node):
    def __init__(self, topics):
        super().__init__('simple_bag_recorder')

        self.writer = rosbag2_py.SequentialWriter()

        storage_options = rosbag2_py._storage.StorageOptions(
            uri='my_bag_7',
            storage_id='mcap'  # Use 'mcap' for MCAP format
        )
        converter_options = rosbag2_py._storage.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)

        self.subs= []

        # Query ROS graph for topic types
        all_topics = dict(self.get_topic_names_and_types())

        for topic_name in topics:
            if topic_name not in all_topics:
                self.get_logger().warn(f"Topic '{topic_name}' not currently available.")
                continue

            # Take the first advertised type if multiple
            msg_type_str = all_topics[topic_name][0]
            msg_module = get_message(msg_type_str)

            topic_info = rosbag2_py._storage.TopicMetadata(
                name=topic_name,
                type=msg_type_str,
                serialization_format='cdr'
            )
            self.writer.create_topic(topic_info)

            sub = self.create_subscription(
                msg_module,
                topic_name,
                lambda msg, t=topic_name: self.topic_callback(msg, t),
                10
            )
            self.subs.append(sub)

            self.get_logger().info(f"Recording topic: {topic_name} [{msg_type_str}]")

    def topic_callback(self, msg, topic_name):
        self.writer.write(
            topic_name,
            serialize_message(msg),
            self.get_clock().now().nanoseconds
        )


def main(args=None):
    rclpy.init(args=args)

    # Example: pass topic names via command line
    # e.g. python record.py chatter numbers
    topics_to_record = sys.argv[1:] if len(sys.argv) > 1 else ['chatter']

    recorder = SimpleBagRecorder(topics_to_record)
    rclpy.spin(recorder)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
