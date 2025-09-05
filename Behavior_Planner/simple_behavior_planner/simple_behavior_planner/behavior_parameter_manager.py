#!/usr/bin/env python3
"""
Behavior Parameter Manager
Manages SMPPI parameters based on behavior types with baseline + percentage system
"""

import yaml
import os
from typing import Dict, Any, Optional
from rclpy.logging import get_logger
from ament_index_python.packages import get_package_share_directory


class BehaviorParameterManager:
    """
    Manages behavior-based parameter modifications for SMPPI controller
    Uses baseline parameters from SMPPI config + percentage modifiers
    """
    
    def __init__(self, smppi_config_path: str, behavior_modifiers_config: dict):
        """
        Initialize parameter manager
        
        Args:
            smppi_config_path: Path to smppi_params.yaml (baseline)
            behavior_modifiers_config: Behavior modifiers configuration
        """
        self.logger = get_logger('BehaviorParameterManager')
        
        # Load baseline parameters from SMPPI config
        self.baseline_params = self._load_smppi_baseline(smppi_config_path)
        
        # Load behavior modifiers
        self.behavior_modifiers = behavior_modifiers_config.get('behavior_modifiers', {})
        
        self.logger.info(f"Loaded baseline params from: {smppi_config_path}")
        self.logger.info(f"Loaded {len(self.behavior_modifiers)} behavior modifiers")
        
        # Log baseline values for debugging
        self._log_baseline_params()
    
    def _load_smppi_baseline(self, config_path: str) -> Dict[str, Any]:
        """Load baseline parameters from SMPPI config file"""
        try:
            # Try to resolve package-relative path first
            resolved_path = self._resolve_package_path(config_path)
            
            if not os.path.exists(resolved_path):
                self.logger.error(f"SMPPI config file not found: {resolved_path} (original: {config_path})")
                return self._get_default_baseline()
            
            with open(resolved_path, 'r') as file:
                config = yaml.safe_load(file)
                
            self.logger.info(f"Successfully loaded SMPPI config from: {resolved_path}")
            
            # Extract relevant parameters from SMPPI config structure
            ros_params = config.get('/**', {}).get('ros__parameters', {})
            
            baseline = {}
            
            # Vehicle parameters
            vehicle = ros_params.get('vehicle', {})
            baseline.update({
                'max_linear_velocity': vehicle.get('max_linear_velocity', 3.0),
                'min_linear_velocity': vehicle.get('min_linear_velocity', 0.0),
                'max_angular_velocity': vehicle.get('max_angular_velocity', 1.16),
                'min_angular_velocity': vehicle.get('min_angular_velocity', -1.16),
                'wheelbase': vehicle.get('wheelbase', 0.65),
                'max_steering_angle': vehicle.get('max_steering_angle', 0.3665),
                'radius': vehicle.get('radius', 0.6),
                'footprint_padding': vehicle.get('footprint_padding', 0.15)
            })
            
            # Cost parameters
            costs = ros_params.get('costs', {})
            baseline.update({
                'obstacle_weight': costs.get('obstacle_weight', 100.0),
                'goal_weight': costs.get('goal_weight', 30.0)
            })
            
            # Lookahead parameters
            lookahead = costs.get('lookahead', {})
            baseline.update({
                'lookahead_base_distance': lookahead.get('base_distance', 1.0),
                'lookahead_velocity_factor': lookahead.get('velocity_factor', 0.4),
                'lookahead_min_distance': lookahead.get('min_distance', 1.0),
                'lookahead_max_distance': lookahead.get('max_distance', 6.0)
            })
            
            # Control parameters
            baseline.update({
                'goal_reached_threshold': ros_params.get('goal_reached_threshold', 2.0),
                'control_frequency': ros_params.get('control_frequency', 20.0)
            })
            
            # Optimizer parameters
            optimizer = ros_params.get('optimizer', {})
            baseline.update({
                'batch_size': optimizer.get('batch_size', 3000),
                'time_steps': optimizer.get('time_steps', 30),
                'model_dt': optimizer.get('model_dt', 0.1),
                'temperature': optimizer.get('temperature', 1.8),
                'lambda_action': optimizer.get('lambda_action', 0.08)
            })
            
            return baseline
            
        except Exception as e:
            self.logger.error(f"Failed to load SMPPI config: {e}")
            return self._get_default_baseline()
    
    def _resolve_package_path(self, config_path: str) -> str:
        """Resolve config path using package share directory"""
        try:
            # If it's already an absolute path, return as is
            if os.path.isabs(config_path):
                return config_path
            
            # Check if it's a package://package_name/path format
            if config_path.startswith('package://'):
                parts = config_path[10:].split('/', 1)  # Remove 'package://'
                if len(parts) == 2:
                    package_name, relative_path = parts
                    package_share_dir = get_package_share_directory(package_name)
                    return os.path.join(package_share_dir, relative_path)
                else:
                    self.logger.error(f"Invalid package path format: {config_path}")
                    return config_path
            
            # If it's just a filename, look in simple_behavior_planner config directory
            if not os.path.dirname(config_path):  # Just a filename
                try:
                    package_share_dir = get_package_share_directory('simple_behavior_planner')
                    candidate_path = os.path.join(package_share_dir, 'config', config_path)
                    if os.path.exists(candidate_path):
                        self.logger.info(f"Found config in simple_behavior_planner: {candidate_path}")
                        return candidate_path
                except Exception:
                    pass
            
            # Try common packages for SMPPI config
            for package_name in ['smppi', 'smppi_controller']:
                try:
                    package_share_dir = get_package_share_directory(package_name)
                    candidate_path = os.path.join(package_share_dir, 'config', config_path)
                    if os.path.exists(candidate_path):
                        self.logger.info(f"Found SMPPI config in package {package_name}: {candidate_path}")
                        return candidate_path
                except Exception:
                    continue  # Package not found, try next
            
            # Fallback: return original path
            return config_path
            
        except Exception as e:
            self.logger.warn(f"Failed to resolve package path for {config_path}: {e}")
            return config_path
    
    def _get_default_baseline(self) -> Dict[str, Any]:
        """Get default baseline parameters if config loading fails"""
        return {
            'max_linear_velocity': 3.0,
            'min_linear_velocity': 0.0,
            'max_angular_velocity': 1.16,
            'min_angular_velocity': -1.16,
            'wheelbase': 0.65,
            'max_steering_angle': 0.3665,
            'radius': 0.6,
            'footprint_padding': 0.15,
            'obstacle_weight': 100.0,
            'goal_weight': 30.0,
            'lookahead_base_distance': 1.0,
            'lookahead_velocity_factor': 0.4,
            'lookahead_min_distance': 1.0,
            'lookahead_max_distance': 6.0,
            'goal_reached_threshold': 2.0,
            'control_frequency': 20.0,
            'xy_goal_tolerance': 2.0,
            'yaw_goal_tolerance': 0.25
        }
    
    def _log_baseline_params(self):
        """Log baseline parameters for debugging"""
        self.logger.info("=== Baseline Parameters ===")
        for key, value in self.baseline_params.items():
            self.logger.info(f"  {key}: {value}")
    
    def get_behavior_params(self, node_type: int) -> Dict[str, Any]:
        """
        Calculate behavior-specific parameters
        
        Args:
            node_type: Node type from MapNode.msg
            
        Returns:
            Dictionary of calculated parameters
        """
        if node_type not in self.behavior_modifiers:
            self.logger.warn(f"Unknown behavior type: {node_type}, using default (type 1)")
            node_type = 1
        
        modifier = self.behavior_modifiers.get(node_type, {})
        multipliers = modifier.get('multipliers', {})
        overrides = modifier.get('overrides', {})
        
        # DEBUG: Log loaded behavior modifier for the requested type
        self.logger.info(f"=== DEBUG: Loaded behavior modifier for type {node_type} ===")
        self.logger.info(f"  Description: {modifier.get('description', 'No description')}")
        self.logger.info(f"  Multipliers: {multipliers}")
        self.logger.info(f"  Overrides: {overrides}")
        
        # Start with baseline parameters
        calculated_params = self.baseline_params.copy()
        
        # DEBUG: Log baseline velocity values
        self.logger.info(f"  Baseline max_linear_velocity: {self.baseline_params.get('max_linear_velocity', 'NOT_FOUND')}")
        self.logger.info(f"  Baseline min_linear_velocity: {self.baseline_params.get('min_linear_velocity', 'NOT_FOUND')}")
        
        # Apply multipliers
        for param_name, multiplier in multipliers.items():
            if param_name in calculated_params:
                baseline_value = self.baseline_params[param_name]
                calculated_params[param_name] = baseline_value * multiplier
                
                self.logger.info(f"  Applied multiplier {param_name}: {baseline_value} * {multiplier} = {calculated_params[param_name]}")
        
        # Apply overrides (absolute values)
        for param_name, override_value in overrides.items():
            old_value = calculated_params.get(param_name, 'NOT_FOUND')
            calculated_params[param_name] = override_value
            self.logger.info(f"  Applied override {param_name}: {old_value} -> {override_value}")
        
        # DEBUG: Log final velocity values  
        self.logger.info(f"=== FINAL CALCULATED VALUES ===")
        self.logger.info(f"  Final max_linear_velocity: {calculated_params.get('max_linear_velocity', 'NOT_FOUND')}")
        self.logger.info(f"  Final min_linear_velocity: {calculated_params.get('min_linear_velocity', 'NOT_FOUND')}")
        
        # Add behavior-specific metadata
        calculated_params['behavior_type'] = node_type
        calculated_params['behavior_description'] = modifier.get('description', f'Behavior {node_type}')
        
        # Add pause duration if specified
        if 'pause_duration' in modifier:
            calculated_params['pause_duration'] = modifier['pause_duration']
        
        return calculated_params
    
    def is_pause_behavior(self, node_type: int) -> bool:
        """Check if behavior type is a pause behavior"""
        return node_type in [7, 8]
    
    def get_pause_duration(self, node_type: int) -> float:
        """Get pause duration for pause behaviors"""
        if not self.is_pause_behavior(node_type):
            return 0.0
        
        modifier = self.behavior_modifiers.get(node_type, {})
        return modifier.get('pause_duration', 1.0)
    
    def get_behavior_description(self, node_type: int) -> str:
        """Get human-readable description of behavior"""
        modifier = self.behavior_modifiers.get(node_type, {})
        return modifier.get('description', f'Unknown behavior {node_type}')
    
    def update_baseline_param(self, param_name: str, value: Any):
        """Update a baseline parameter (useful for runtime adjustments)"""
        if param_name in self.baseline_params:
            old_value = self.baseline_params[param_name]
            self.baseline_params[param_name] = value
            self.logger.info(f"Updated baseline {param_name}: {old_value} -> {value}")
        else:
            self.logger.warn(f"Unknown baseline parameter: {param_name}")
    
    def get_available_behaviors(self) -> Dict[int, str]:
        """Get list of available behavior types and their descriptions"""
        behaviors = {}
        for node_type, modifier in self.behavior_modifiers.items():
            behaviors[node_type] = modifier.get('description', f'Behavior {node_type}')
        return behaviors
    
    def validate_behavior_params(self, params: Dict[str, Any]) -> bool:
        """Validate calculated parameters for safety"""
        try:
            # Check velocity limits
            max_vel = params.get('max_linear_velocity', 0.0)
            min_vel = params.get('min_linear_velocity', 0.0)
            
            if max_vel < 0 and min_vel >= 0:
                self.logger.error(f"Invalid velocity config: max={max_vel}, min={min_vel}")
                return False
            
            # Check goal weight is positive
            goal_weight = params.get('goal_weight', 1.0)
            if goal_weight <= 0:
                self.logger.error(f"Invalid goal_weight: {goal_weight}")
                return False
            
            # Check lookahead distances
            lookahead_min = params.get('lookahead_min_distance', 0.0)
            lookahead_max = params.get('lookahead_max_distance', 1.0)
            
            if lookahead_min >= lookahead_max:
                self.logger.error(f"Invalid lookahead: min={lookahead_min}, max={lookahead_max}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Parameter validation error: {e}")
            return False