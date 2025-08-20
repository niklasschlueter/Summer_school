import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
import tf2_ros
import pinocchio as pin
import numpy as np

class EEFrameBroadcaster(Node):
    def __init__(self, model, data, ee_frame_name="tool0"):
        super().__init__('ee_tf_broadcaster')
        self.model = model
        self.data = data
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id = model.getFrameId(ee_frame_name)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_ee_frame)  # 10 Hz

        # Example joint configuration
        self.q = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])

    def publish_ee_frame(self):
        # Forward kinematics
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)

        placement = self.data.oMf[self.ee_frame_id]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"   # make sure matches your robot base frame
        t.child_frame_id = "tool0_pin" #self.ee_frame_name

        # Translation
        t.transform.translation.x = placement.translation[0]
        t.transform.translation.y = placement.translation[1]
        t.transform.translation.z = placement.translation[2]

        # Rotation (quaternion)
        quat = pin.Quaternion(placement.rotation).coeffs()  # x,y,z,w
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    model, _, _ = pin.buildModelsFromUrdf("ur5e.urdf")
    data = model.createData()

    node = EEFrameBroadcaster(model, data)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
