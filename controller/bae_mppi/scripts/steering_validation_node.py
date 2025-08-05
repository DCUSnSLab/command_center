#!/usr/bin/env python3
"""
Steering Validation Node for bae_mppi
Compares MPPI commanded steering vs actual /odom feedback to detect discrepancies
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import time
from collections import deque

# ROS2 messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import json

class SteeringValidationNode(Node):
    """Validates steering commands vs actual robot response"""
    
    def __init__(self):
        super().__init__('steering_validation')
        
        # Topic parameters
        self.declare_parameter('topics.input.cmd_vel_monitor', '/ackermann_like_controller/cmd_vel')
        self.declare_parameter('topics.input.odometry', '/odom')
        self.declare_parameter('topics.output.steering_validation', '/steering_validation')
        
        # Parameters
        self.declare_parameter('wheelbase', 0.65)
        self.declare_parameter('validation_window', 5.0)  # seconds
        self.declare_parameter('log_interval', 2.0)  # seconds
        
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.validation_window = self.get_parameter('validation_window').get_parameter_value().double_value
        self.log_interval = self.get_parameter('log_interval').get_parameter_value().double_value
        
        # Get topic names
        cmd_vel_topic = self.get_parameter('topics.input.cmd_vel_monitor').get_parameter_value().string_value
        odom_topic = self.get_parameter('topics.input.odometry').get_parameter_value().string_value
        validation_topic = self.get_parameter('topics.output.steering_validation').get_parameter_value().string_value
        
        # Data storage
        self.cmd_history = deque(maxlen=int(self.validation_window * 20))  # 20Hz assumption
        self.odom_history = deque(maxlen=int(self.validation_window * 50))  # Higher frequency
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, 
            self.cmd_vel_callback, reliable_qos)
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, 
            self.odom_callback, reliable_qos)
        
        # Publishers
        self.validation_pub = self.create_publisher(
            String, validation_topic, reliable_qos)
        
        # Validation timer
        self.validation_timer = self.create_timer(
            self.log_interval, self.validation_callback)
        
        self.get_logger().info('Steering Validation Node initialized')
        self.get_logger().info(f'Wheelbase: {self.wheelbase}m, Window: {self.validation_window}s')
    
    def cmd_vel_callback(self, msg: Twist):
        """Store commanded velocities with timestamp"""
        current_time = time.time()
        
        # Convert cmd_vel to expected steering angle
        if abs(msg.linear.x) > 0.01:
            # Reverse the angular velocity calculation from mppi_core_node.py
            # angular_velocity = (velocity / wheelbase) * tan(steering_angle)
            # Therefore: tan(steering_angle) = angular_velocity * wheelbase / velocity
            expected_steering = np.arctan(msg.angular.z * self.wheelbase / msg.linear.x)
        else:
            expected_steering = 0.0
        
        cmd_data = {
            'timestamp': current_time,
            'linear_vel': msg.linear.x,
            'angular_vel': msg.angular.z,
            'expected_steering': expected_steering
        }
        
        self.cmd_history.append(cmd_data)
    
    def odom_callback(self, msg: Odometry):
        """Store actual odometry data with timestamp"""
        current_time = time.time()
        
        # Extract velocity from odometry
        linear_vel = msg.twist.twist.linear.x
        angular_vel = msg.twist.twist.angular.z
        
        # Calculate actual steering angle from odometry
        if abs(linear_vel) > 0.01:
            actual_steering = np.arctan(angular_vel * self.wheelbase / linear_vel)
        else:
            actual_steering = 0.0
        
        odom_data = {
            'timestamp': current_time,
            'actual_linear_vel': linear_vel,
            'actual_angular_vel': angular_vel,
            'actual_steering': actual_steering,
            'position_x': msg.pose.pose.position.x,
            'position_y': msg.pose.pose.position.y
        }
        
        self.odom_history.append(odom_data)
    
    def validation_callback(self):
        """Perform validation analysis"""
        if len(self.cmd_history) < 5 or len(self.odom_history) < 5:
            return
        
        current_time = time.time()
        
        # Filter recent data within validation window
        recent_cmds = [cmd for cmd in self.cmd_history 
                      if current_time - cmd['timestamp'] <= self.validation_window]
        recent_odom = [odom for odom in self.odom_history 
                      if current_time - odom['timestamp'] <= self.validation_window]
        
        if len(recent_cmds) < 3 or len(recent_odom) < 3:
            return
        
        # Analyze steering discrepancies
        steering_errors = []
        velocity_errors = []
        matched_pairs = []
        
        for cmd in recent_cmds:
            # Find odom measurement that accounts for robot response delay
            closest_odom = None
            min_time_diff = float('inf')
            
            for odom in recent_odom:
                # Look for odom data 100-500ms AFTER command (robot response delay)
                time_diff = odom['timestamp'] - cmd['timestamp']
                if 0.1 <= time_diff <= 0.5 and time_diff < min_time_diff:  # 100-500ms delay
                    min_time_diff = time_diff
                    closest_odom = odom
            
            if closest_odom:
                steering_error = abs(cmd['expected_steering'] - closest_odom['actual_steering'])
                velocity_error = abs(cmd['linear_vel'] - closest_odom['actual_linear_vel'])
                
                steering_errors.append(steering_error)
                velocity_errors.append(velocity_error)
                matched_pairs.append({
                    'cmd_steering': cmd['expected_steering'],
                    'actual_steering': closest_odom['actual_steering'],
                    'cmd_velocity': cmd['linear_vel'],
                    'actual_velocity': closest_odom['actual_linear_vel'],
                    'time_diff': min_time_diff
                })
        
        if steering_errors:
            # Calculate statistics
            avg_steering_error = np.mean(steering_errors)
            max_steering_error = np.max(steering_errors)
            avg_velocity_error = np.mean(velocity_errors)
            
            # Check for significant discrepancies (more lenient thresholds)
            significant_steering_error = avg_steering_error > 0.2  # 0.2 rad ≈ 11.5°
            significant_velocity_error = avg_velocity_error > 0.2  # 0.2 m/s
            
            # Prepare validation report
            report = {
                'timestamp': current_time,
                'wheelbase_used': self.wheelbase,
                'sample_count': len(matched_pairs),
                'avg_steering_error_rad': float(avg_steering_error),
                'avg_steering_error_deg': float(np.degrees(avg_steering_error)),
                'max_steering_error_rad': float(max_steering_error),
                'max_steering_error_deg': float(np.degrees(max_steering_error)),
                'avg_velocity_error': float(avg_velocity_error),
                'significant_steering_discrepancy': bool(significant_steering_error),
                'significant_velocity_discrepancy': bool(significant_velocity_error),
                'recent_samples': matched_pairs[-3:] if len(matched_pairs) >= 3 else matched_pairs
            }
            
            # Log findings
            if significant_steering_error or significant_velocity_error:
                self.get_logger().warn(
                    f'STEERING DISCREPANCY: Avg={np.degrees(avg_steering_error):.2f}°, '
                    f'Max={np.degrees(max_steering_error):.2f}°, VelErr={avg_velocity_error:.3f}m/s'
                )
            else:
                self.get_logger().info(
                    f'Steering OK: Avg={np.degrees(avg_steering_error):.2f}°, '
                    f'Samples={len(matched_pairs)}'
                )
            
            # Publish validation report
            self.validation_pub.publish(String(data=json.dumps(report, indent=2)))
        
        # Parameter recommendations (only analyze when we have enough data)
        if len(matched_pairs) > 20:  # Increased threshold for more reliable analysis
            # Only run parameter analysis every 10 seconds to reduce noise
            if not hasattr(self, '_last_param_analysis') or (current_time - self._last_param_analysis) > 10.0:
                self.analyze_parameter_recommendations(matched_pairs)
                self._last_param_analysis = current_time
    
    def analyze_parameter_recommendations(self, matched_pairs):
        """Analyze data to recommend parameter adjustments"""
        # Calculate apparent wheelbase from actual robot behavior
        valid_measurements = []
        
        for pair in matched_pairs:
            # Only analyze when robot is moving with significant steering
            if (abs(pair['actual_velocity']) > 0.05 and 
                abs(pair['actual_steering']) > 0.02):  # ~1.15 degrees
                
                # From tricycle model: angular_vel = (v / wheelbase) * tan(steering)
                # Therefore: wheelbase = v * tan(steering) / angular_vel
                if abs(pair['actual_velocity']) > 0.01:
                    # Calculate expected angular velocity from commanded steering
                    expected_angular_vel = (pair['actual_velocity'] / self.wheelbase) * np.tan(pair['cmd_steering'])
                    
                    # Calculate what wheelbase would give the observed behavior
                    if abs(expected_angular_vel) > 0.01:
                        # From actual measurements: wheelbase = v * tan(actual_steering) / actual_angular_vel
                        actual_angular_vel = pair['actual_velocity'] * np.tan(pair['actual_steering']) / self.wheelbase
                        
                        # More robust estimation using commanded vs actual relationship
                        if abs(pair['cmd_steering']) > 0.01:
                            # Scale factor between commanded and actual steering
                            steering_ratio = pair['actual_steering'] / pair['cmd_steering']
                            
                            # If steering_ratio is consistent, it suggests wheelbase issue
                            if 0.5 < steering_ratio < 2.0:  # Reasonable range
                                apparent_wheelbase = self.wheelbase / steering_ratio
                                
                                if 0.3 < apparent_wheelbase < 1.2:  # Physical limits
                                    valid_measurements.append(apparent_wheelbase)
        
        if len(valid_measurements) > 8:  # Need more samples for reliability
            # Use median to reduce outlier effects
            estimated_wheelbase = np.median(valid_measurements)
            measurement_std = np.std(valid_measurements)
            wheelbase_error = abs(estimated_wheelbase - self.wheelbase)
            
            # Only recommend change if error is significant and measurements are consistent
            if wheelbase_error > 0.08 and measurement_std < 0.15:  # 8cm error, 15cm std
                self.get_logger().warn(
                    f'WHEELBASE DISCREPANCY: Config={self.wheelbase:.3f}m, '
                    f'Estimated={estimated_wheelbase:.3f}m, Error={wheelbase_error:.3f}m'
                )
                self.get_logger().info(
                    f'RECOMMENDATION: Update wheelbase parameter to {estimated_wheelbase:.3f}m'
                )


def main(args=None):
    rclpy.init(args=args)
    
    node = SteeringValidationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()