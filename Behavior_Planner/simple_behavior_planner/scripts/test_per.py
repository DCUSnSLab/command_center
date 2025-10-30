#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
import sys
import termios
import tty
import threading
import time


class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')

        # Publishers
        self.state_pub = self.create_publisher(Int32, '/tl/state_id', 10)
        self.path_pub = self.create_publisher(Bool, '/path_availability', 10)

        # 현재 상태 값 (기본 0)
        self.current_state = 0

        # 5Hz 타이머 (0.2초 주기)
        self.create_timer(0.2, self.publish_state)

        # 키 입력 스레드 실행
        self.key_thread = threading.Thread(target=self.key_loop, daemon=True)
        self.key_thread.start()

    def publish_state(self):
        """마지막 누른 값 계속 퍼블리시"""
        msg = Int32()
        msg.data = self.current_state
        self.state_pub.publish(msg)

    def key_loop(self):
        """키보드 입력 처리 루프"""
        old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            while rclpy.ok():
                key = sys.stdin.read(1)

                if key in ['1', '2', '3', '4']:
                    self.current_state = int(key)
                    self.get_logger().info(f"State set to {self.current_state}")

                elif key == 'q':
                    msg = Bool()
                    msg.data = True
                    self.path_pub.publish(msg)
                    self.get_logger().info("Published /path_availability: True")

                elif key == 'w':
                    msg = Bool()
                    msg.data = False
                    self.path_pub.publish(msg)
                    self.get_logger().info("Published /path_availability: False")

                elif key == '\x03':  # Ctrl+C
                    break
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
