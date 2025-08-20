import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class URStopClient(Node):
    def __init__(self):
        super().__init__('ur_stop_client')
        self.cli = self.create_client(Trigger, '/dashboard_client/stop')

        # wait for service
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /dashboard_client/stop service...')

        self.req = Trigger.Request()

    def send_request(self):
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    ur_stop_client = URStopClient()
    response = ur_stop_client.send_request()

    if response.success:
        ur_stop_client.get_logger().info('Robot stopped successfully!')
    else:
        ur_stop_client.get_logger().warn(f'Failed to stop robot: {response.message}')

    ur_stop_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
