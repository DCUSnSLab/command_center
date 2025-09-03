#!/usr/bin/env python3
"""
Test script for behavior parameter system
Tests the BehaviorParameterManager integration
"""

import rclpy
from rclpy.node import Node
import os
import yaml
import sys

# Add the package path to sys.path
package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(package_path, 'simple_behavior_planner'))

from behavior_parameter_manager import BehaviorParameterManager


class BehaviorParameterTester(Node):
    def __init__(self):
        super().__init__('behavior_parameter_tester')
        
        self.get_logger().info("Behavior Parameter Tester initialized")
        
        # Load config and test parameter manager
        self.test_parameter_manager()
    
    def test_parameter_manager(self):
        """Test the BehaviorParameterManager"""
        try:
            # Get config paths
            package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(package_path, 'config', 'behavior_modifiers.yaml')
            
            if not os.path.exists(config_path):
                self.get_logger().error(f"Config file not found: {config_path}")
                return
            
            # Load behavior modifiers config
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            behavior_config = config.get('/**', {}).get('ros__parameters', {})
            smppi_config_path = behavior_config.get('smppi_config_path', 
                '/home/d2-521-30/repo/command_center_ws/src/command_center/controller/smppi/config/smppi_params.yaml')
            
            self.get_logger().info(f"Loading SMPPI config from: {smppi_config_path}")
            
            # Initialize parameter manager
            param_manager = BehaviorParameterManager(smppi_config_path, behavior_config)
            
            # Test each behavior type
            self.get_logger().info("\\n=== Testing Behavior Parameters ===")
            
            behaviors = param_manager.get_available_behaviors()
            for behavior_type, description in behaviors.items():
                self.get_logger().info(f"\\n--- Testing Behavior {behavior_type}: {description} ---")
                
                # Get calculated parameters
                params = param_manager.get_behavior_params(behavior_type)
                
                # Validate parameters
                is_valid = param_manager.validate_behavior_params(params)
                
                self.get_logger().info(f"Parameters valid: {is_valid}")
                
                # Show key parameters
                key_params = ['max_linear_velocity', 'min_linear_velocity', 'goal_weight', 
                             'lookahead_base_distance', 'xy_goal_tolerance']
                
                for key in key_params:
                    if key in params:
                        self.get_logger().info(f"  {key}: {params[key]}")
                
                # Check pause behaviors
                if param_manager.is_pause_behavior(behavior_type):
                    pause_duration = param_manager.get_pause_duration(behavior_type)
                    self.get_logger().info(f"  Pause duration: {pause_duration} seconds")
            
            # Test baseline parameters
            self.get_logger().info("\\n=== Baseline Parameters ===")
            baseline = param_manager.baseline_params
            for key, value in baseline.items():
                self.get_logger().info(f"  {key}: {value}")
            
            self.get_logger().info("\\n=== Test completed successfully ===")
            
        except Exception as e:
            self.get_logger().error(f"Test failed: {e}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")


def main():
    rclpy.init()
    
    tester = BehaviorParameterTester()
    
    try:
        # Just run the test once
        rclpy.spin_once(tester, timeout_sec=1.0)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()