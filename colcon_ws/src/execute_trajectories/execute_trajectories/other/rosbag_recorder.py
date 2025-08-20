import subprocess
import os
from datetime import datetime

class RosbagRecorder:
    def __init__(self, topics, output_dir="."):
        """
        :param topics: list of topics to record (e.g. ["/joint_states", "/tf"])
        :param output_dir: directory where the bag will be stored
        """
        self.topics = topics
        self.output_dir = os.path.abspath(output_dir)
        self.process = None

    def start(self, bag_name=None):
        """Start rosbag recording"""
        if self.process is not None:
            print("Recording already in progress!")
            return

        if bag_name is None:
            bag_name = f"rosbag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        bag_path = os.path.join(self.output_dir, bag_name)

        # Build command
        cmd = ["ros2", "bag", "record", "-o", bag_path] + self.topics

        # Launch recording process
        self.process = subprocess.Popen(cmd)
        print(f"✅ Started recording topics {self.topics} into {bag_path}")

    def stop(self):
        """Stop rosbag recording"""
        if self.process is None:
            print("⚠️ No recording is running.")
            return

        self.process.terminate()
        self.process.wait()
        print("🛑 Recording stopped.")

        self.process = None

if __name__ == "__main__":
    recorder = RosbagRecorder(["/joint_states", "/tf"], output_dir="bags")

    # Start recording
    recorder.start("my_experiment")

    import time
    time.sleep(5)  # record for 5 seconds

    # Stop recording
    recorder.stop()