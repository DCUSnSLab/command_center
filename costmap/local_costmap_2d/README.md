# Local Costmap 2D (Python)

A lightweight, PointCloud-based local costmap package for ROS2 that provides obstacle detection and RViz visualization without nav2 dependencies. **This is a pure Python implementation.**

## Features

- **PointCloud Processing**: Supports various PointCloud2 sources (Velodyne, RGB-D cameras, etc.)
- **Advanced Filtering**: Ground plane removal (RANSAC), statistical outlier filtering, voxel grid downsampling
- **Real-time Performance**: Optimized for 20Hz processing with temporal decay for dynamic obstacles
- **RViz Visualization**: Publishes OccupancyGrid and obstacle markers for visualization
- **Configurable**: Extensive parameter configuration for different robot platforms
- **No Nav2 Dependencies**: Lightweight alternative to nav2_costmap_2d
- **Pure Python**: Easy to understand, modify, and extend

## Quick Start

### Building

```bash
cd /home/d2-521-30/repo/command_center_ws
colcon build --packages-select local_costmap_2d
source install/setup.bash
```

### Running

```bash
# Basic usage
ros2 launch local_costmap_2d local_costmap.launch.py

# With custom parameters
ros2 launch local_costmap_2d local_costmap.launch.py params_file:=/path/to/your/params.yaml

# Enable debug logging
ros2 launch local_costmap_2d local_costmap.launch.py log_level:=debug
```

## Topics

### Subscribed Topics

- `/points` (sensor_msgs/PointCloud2) - Input point cloud data
- `/odom` (nav_msgs/Odometry) - Robot odometry for map centering

### Published Topics

- `/local_costmap/costmap` (nav_msgs/OccupancyGrid) - Main costmap for navigation
- `/local_costmap/costmap_updates` (map_msgs/OccupancyGridUpdate) - Incremental updates
- `/local_costmap/obstacle_markers` (visualization_msgs/MarkerArray) - Obstacle visualization
- `/local_costmap/filtered_pointcloud` (sensor_msgs/PointCloud2) - Filtered points for debugging

## Parameters

### Basic Costmap Settings

- `resolution` (double, default: 0.05): Grid resolution in meters per cell
- `width` (double, default: 20.0): Costmap width in meters
- `height` (double, default: 20.0): Costmap height in meters
- `publish_frequency` (double, default: 10.0): Publishing frequency in Hz
- `update_frequency` (double, default: 20.0): Processing frequency in Hz

### Frame Configuration

- `global_frame` (string, default: "odom"): Global coordinate frame
- `robot_frame` (string, default: "base_link"): Robot base frame
- `sensor_frame` (string, default: "velodyne"): PointCloud sensor frame

### PointCloud Processing

- `max_range` (double, default: 100.0): Maximum sensor range in meters
- `min_range` (double, default: 0.1): Minimum sensor range in meters
- `min_obstacle_height` (double, default: 0.1): Minimum obstacle height in meters
- `max_obstacle_height` (double, default: 2.0): Maximum obstacle height in meters
- `voxel_size` (double, default: 0.05): Voxel grid downsample size in meters

### Filtering Options

- `enable_ground_removal` (bool, default: true): Enable RANSAC ground plane removal
- `enable_statistical_filter` (bool, default: true): Enable statistical outlier removal
- `enable_temporal_decay` (bool, default: true): Enable temporal decay for dynamic obstacles
- `decay_rate` (double, default: 0.95): Decay rate per frame (0.95 = 5% decay per frame)

### Robot Footprint

- `robot_footprint` (double array): Robot footprint vertices [x1,y1,x2,y2,...]
- `footprint_padding` (double, default: 0.15): Additional padding around robot

## Architecture

### Core Components

1. **SimpleCostmap2D**: Core 2D grid data structure with coordinate transformations (numpy-based)
2. **PointCloudProcessor**: Main processing pipeline with filtering and projection
3. **PointCloudFilters**: Collection of filtering algorithms (RANSAC, statistical, clustering)
4. **CostmapPublisher**: ROS publishing interface with visualization support

### Processing Pipeline

```
PointCloud2 Input
    ↓
Range Filtering
    ↓
Height Filtering
    ↓
Ground Plane Removal (RANSAC)
    ↓
Voxel Grid Downsampling
    ↓
Statistical Outlier Removal
    ↓
Euclidean Clustering (DBSCAN)
    ↓
Robot Footprint Filtering
    ↓
2D Grid Projection
    ↓
Temporal Decay
    ↓
OccupancyGrid Output
```

## Python Dependencies

This package requires the following Python packages:
- numpy
- scipy
- scikit-learn (for DBSCAN clustering)
- rclpy
- sensor_msgs_py
- tf2_ros
- tf2_geometry_msgs
- tf2_sensor_msgs

These are automatically installed when building the package.

## Configuration Examples

### For Velodyne LiDAR

```yaml
local_costmap:
  ros__parameters:
    sensor_frame: "velodyne"
    max_range: 100.0
    min_obstacle_height: 0.2
    max_obstacle_height: 3.0
    enable_ground_removal: true
    voxel_size: 0.1
```

### For RGB-D Camera

```yaml
local_costmap:
  ros__parameters:
    sensor_frame: "camera_depth_optical_frame"
    max_range: 10.0
    min_obstacle_height: 0.1
    max_obstacle_height: 2.0
    voxel_size: 0.02
```

## Performance Tuning

### High Performance Settings

```yaml
# Reduce processing load
voxel_size: 0.1                    # Larger voxels
enable_statistical_filter: false   # Skip statistical filtering
min_cluster_size: 20               # Larger clusters only
update_frequency: 10.0             # Lower processing rate
```

### High Accuracy Settings

```yaml
# Increase accuracy
voxel_size: 0.02                   # Smaller voxels
enable_statistical_filter: true    # Enable all filtering
statistical_mean_k: 100            # More neighbors for analysis
ground_max_iterations: 2000        # More RANSAC iterations
```

## Troubleshooting

### Common Issues

1. **No costmap output**: Check TF frames and ensure `/points` topic is publishing
2. **Poor ground removal**: Adjust `ground_distance_threshold` and `ground_max_iterations`
3. **Too many false obstacles**: Increase `min_cluster_size` or enable statistical filtering
4. **High CPU usage**: Increase `voxel_size` or reduce `update_frequency`

### Debug Topics

Monitor these topics for debugging:

```bash
# Check filtered point cloud
ros2 topic echo /local_costmap/filtered_pointcloud

# Check obstacle markers count
ros2 topic echo /local_costmap/obstacle_markers

# Monitor processing rate
ros2 topic hz /local_costmap/costmap
```

## Integration

### With RViz

Add these displays to RViz:
- Map display for `/local_costmap/costmap`
- MarkerArray display for `/local_costmap/obstacle_markers`
- PointCloud2 display for `/local_costmap/filtered_pointcloud`

### With Navigation Stack

The package publishes standard `nav_msgs/OccupancyGrid` messages compatible with most navigation frameworks.

## File Structure

```
local_costmap_2d/
├── local_costmap_2d/          # Python package
│   ├── __init__.py
│   ├── cost_values.py         # Cost constants
│   ├── costmap_2d.py          # Core costmap class
│   ├── pointcloud_filters.py  # Filtering algorithms
│   ├── pointcloud_processor.py # Processing pipeline
│   ├── costmap_publisher.py   # ROS2 publishers
│   └── local_costmap_node.py  # Main node
├── config/
│   └── costmap_params.yaml    # Default parameters
├── launch/
│   ├── local_costmap.launch.py
│   └── local_costmap_auto.launch.py
├── setup.py                   # Python package setup
├── setup.cfg
├── package.xml
└── README.md

# Backed up C++ files (for reference)
├── include_cpp_backup/        # Original C++ headers
├── src_cpp_backup/           # Original C++ sources
└── CMakeLists.txt.backup     # Original CMake config
```

## Conversion Notes

This package was converted from C++ to Python. Key changes:

- PCL (Point Cloud Library) → NumPy/SciPy
- Euclidean Clustering → DBSCAN (scikit-learn)
- std::vector → Python lists/NumPy arrays
- std::mutex → threading.Lock
- Raw pointers → Python object references

The original C++ code is backed up in `include_cpp_backup/` and `src_cpp_backup/` directories.

## License

BSD-3-Clause
