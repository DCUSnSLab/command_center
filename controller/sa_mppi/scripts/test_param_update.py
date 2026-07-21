#!/usr/bin/env python3
"""
Test script for MPPI dynamic parameter updates
"""

import rclpy
from rclpy.node import Node
from command_center_interfaces.msg import MPPIParams
from std_msgs.msg import Header

class MPPIParamTester(Node):
    def __init__(self):
        super().__init__('mppi_param_tester')
        
        # Create publisher
        self.param_pub = self.create_publisher(
            MPPIParams, '/mppi_update_params', 10
        )
        
        # Wait for publisher to be ready
        self.get_logger().info("MPPI Parameter Tester initialized")
        self.get_logger().info("Available test commands:")
        self.get_logger().info("1. update_lookahead - Update lookahead parameters")
        self.get_logger().info("2. update_goal_critic - Update goal critic parameters")
        self.get_logger().info("3. update_costs - Update cost weights")
        self.get_logger().info("4. update_vehicle - Update vehicle speed limits")
        self.get_logger().info("5. reset_to_defaults - Reset parameters to defaults")
    
    def update_lookahead_params(self):
        """Test lookahead parameter updates"""
        msg = MPPIParams()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Update lookahead parameters
        msg.update_lookahead = True
        msg.lookahead_base_distance = 4.0  # Increased from 3.0
        msg.lookahead_velocity_factor = 1.5  # Increased from 1.2
        msg.lookahead_min_distance = 1.5    # Increased from 1.0
        msg.lookahead_max_distance = 8.0    # Increased from 6.0
        
        self.param_pub.publish(msg)
        self.get_logger().info("Published lookahead parameter updates")
    
    def update_goal_critic_params(self):
        """Test goal critic parameter updates"""
        msg = MPPIParams()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Update goal critic parameters
        msg.update_goal_critic = True
        msg.xy_goal_tolerance = 0.5      # Increased from 0.25
        msg.yaw_goal_tolerance = 0.5     # Increased from 0.25
        msg.distance_scale = 2.0         # Increased from default
        msg.angle_scale = 1.5            # Increased from default
        msg.alignment_scale = 0.8        # Custom value
        msg.progress_scale = 0.1         # Enable progress reward
        msg.use_progress_reward = True
        msg.respect_reverse_heading = False  # Disable reverse heading
        msg.yaw_blend_distance = 2.0     # Increased from 1.5
        
        self.param_pub.publish(msg)
        self.get_logger().info("Published goal critic parameter updates")
    
    def update_cost_weights(self):
        """Test cost weight updates"""
        msg = MPPIParams()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Update cost weights
        msg.update_costs = True
        msg.obstacle_weight = 150.0  # Increased from 100.0
        msg.goal_weight = 50.0       # Increased from 30.0
        
        self.param_pub.publish(msg)
        self.get_logger().info("Published cost weight updates")
    
    def update_vehicle_params(self):
        """Test vehicle speed limit updates"""
        msg = MPPIParams()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        print("\n=== Vehicle Speed Update Options ===")
        print("1. Conservative (Low speed)")
        print("2. Normal (Default speed)")  
        print("3. Aggressive (High speed)")
        print("4. Custom values")
        
        try:
            choice = input("Enter speed profile choice (1-4): ").strip()
            
            msg.update_vehicle = True
            msg.wheelbase = 0.65  # Keep wheelbase constant
            
            if choice == '1':
                # Conservative - Low speed
                msg.max_linear_velocity = 1.0   # Reduced from 2.0
                msg.max_angular_velocity = 0.8  # Reduced from 1.16
                msg.min_angular_velocity = -0.8
                self.get_logger().info("Applied conservative speed limits")
                
            elif choice == '2':
                # Normal - Default values
                msg.max_linear_velocity = 2.0
                msg.max_angular_velocity = 1.16
                msg.min_angular_velocity = -1.16
                self.get_logger().info("Applied normal speed limits")
                
            elif choice == '3':
                # Aggressive - High speed
                msg.max_linear_velocity = 3.0   # Increased from 2.0
                msg.max_angular_velocity = 1.5  # Increased from 1.16
                msg.min_angular_velocity = -1.5
                self.get_logger().info("Applied aggressive speed limits")
                
            elif choice == '4':
                # Custom values
                try:
                    max_linear = float(input("Enter max linear velocity (m/s, current: 2.0): ").strip())
                    max_angular = float(input("Enter max angular velocity (rad/s, current: 1.16): ").strip())
                    
                    msg.max_linear_velocity = max(0.1, max_linear)  # Minimum 0.1 m/s
                    msg.max_angular_velocity = max(0.1, max_angular)  # Minimum 0.1 rad/s
                    msg.min_angular_velocity = -msg.max_angular_velocity
                    
                    self.get_logger().info(f"Applied custom speeds: linear={msg.max_linear_velocity}, angular={msg.max_angular_velocity}")
                    
                except ValueError:
                    self.get_logger().error("Invalid input. Using default values.")
                    msg.max_linear_velocity = 2.0
                    msg.max_angular_velocity = 1.16
                    msg.min_angular_velocity = -1.16
            else:
                self.get_logger().error("Invalid choice. Using default values.")
                msg.max_linear_velocity = 2.0
                msg.max_angular_velocity = 1.16
                msg.min_angular_velocity = -1.16
            
            # Always set min linear velocity to 0
            msg.min_linear_velocity = 0.0
            
            self.param_pub.publish(msg)
            self.get_logger().info(f"Published vehicle parameter updates: "
                                 f"linear=[{msg.min_linear_velocity:.1f}, {msg.max_linear_velocity:.1f}], "
                                 f"angular=[{msg.min_angular_velocity:.2f}, {msg.max_angular_velocity:.2f}]")
            
        except Exception as e:
            self.get_logger().error(f"Error in vehicle parameter update: {e}")
    
    def reset_to_defaults(self):
        """Reset parameters to default values"""
        msg = MPPIParams()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Reset lookahead parameters
        msg.update_lookahead = True
        msg.lookahead_base_distance = 3.0
        msg.lookahead_velocity_factor = 1.2
        msg.lookahead_min_distance = 1.0
        msg.lookahead_max_distance = 6.0
        
        # Reset goal critic parameters
        msg.update_goal_critic = True
        msg.xy_goal_tolerance = 0.25
        msg.yaw_goal_tolerance = 0.25
        msg.distance_scale = 1.0
        msg.angle_scale = 1.0
        msg.alignment_scale = 1.0
        msg.progress_scale = 0.0
        msg.use_progress_reward = False
        msg.respect_reverse_heading = True
        msg.yaw_blend_distance = 1.5
        
        # Reset cost weights
        msg.update_costs = True
        msg.obstacle_weight = 100.0
        msg.goal_weight = 30.0
        
        # Reset vehicle parameters
        msg.update_vehicle = True
        msg.wheelbase = 0.65
        msg.max_linear_velocity = 2.0
        msg.min_linear_velocity = 0.0
        msg.max_angular_velocity = 1.16
        msg.min_angular_velocity = -1.16
        
        self.param_pub.publish(msg)
        self.get_logger().info("Published parameter reset to defaults")

def main():
    rclpy.init()
    tester = MPPIParamTester()
    
    # Interactive menu
    while rclpy.ok():
        print("\n=== MPPI Parameter Tester ===")
        print("1. Update lookahead parameters")
        print("2. Update goal critic parameters") 
        print("3. Update cost weights")
        print("4. Update vehicle speed limits")
        print("5. Reset to defaults")
        print("6. Exit")
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                tester.update_lookahead_params()
            elif choice == '2':
                tester.update_goal_critic_params()
            elif choice == '3':
                tester.update_cost_weights()
            elif choice == '4':
                tester.update_vehicle_params()
            elif choice == '5':
                tester.reset_to_defaults()
            elif choice == '6':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()