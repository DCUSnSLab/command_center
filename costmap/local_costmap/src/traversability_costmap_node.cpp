// /costmap from the traversability grid (SCV_Perception/traversability_grid).
//
// No accumulation here: the grid already carries its own memory. Each grid message
// is turned into one /costmap:
//   - unobserved (-1) cells become unknown_cost,
//   - the robot footprint is cleared,
//   - the map semantic layer is sampled into /costmap/keepout on the same window.
// If the grid stops for grid_timeout, /costmap goes all UNKNOWN (smppi: lethal).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "local_costmap/costmap_2d.hpp"
#include "local_costmap/semantic_layer.hpp"

using nav_msgs::msg::OccupancyGrid;

namespace local_costmap
{

namespace
{
double yawOf(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

// Costmap2D with the grid message's cell layout. Its constructor rounds the size up,
// and cells * res can land just above a whole cell count, so pass cells - 0.5.
void fitTo(std::unique_ptr<Costmap2D> & m, const OccupancyGrid & g)
{
  const int w = static_cast<int>(g.info.width), h = static_cast<int>(g.info.height);
  const double res = g.info.resolution;
  if (!m || m->getWidthCells() != w || m->getHeightCells() != h || m->getResolution() != res) {
    m = std::make_unique<Costmap2D>((w - 0.5) * res, (h - 0.5) * res, res);
  }
  m->updateOrigin(g.info.origin.position.x, g.info.origin.position.y);
}
}  // namespace

class TraversabilityCostmapNode : public rclcpp::Node
{
public:
  TraversabilityCostmapNode()
  : Node("traversability_costmap_node")
  {
    const auto grid_topic = declare_parameter<std::string>(
      "grid_topic", "/perception/traversability/grid");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    grid_timeout_ = declare_parameter<double>("grid_timeout", 1.0);
    unknown_cost_ = static_cast<int8_t>(declare_parameter<int>("unknown_cost", 0));
    footprint_ = declare_parameter<std::vector<double>>(
      "robot_footprint", {0.49, 0.3725, 0.49, -0.3725, -0.49, -0.3725, -0.49, 0.3725});
    const bool use_semantic = declare_parameter<bool>("use_semantic_layer", true);
    semantic_topic_ = declare_parameter<std::string>(
      "semantic_grid_topic", "/map_provider_node/semantic/grid");
    const int crosswalk_cost = static_cast<int>(declare_parameter<int>("semantic_crosswalk_cost", 0));

    if (footprint_.size() < 6 || footprint_.size() % 2 != 0) {
      throw std::runtime_error("robot_footprint must be [x1, y1, x2, y2, ...] with >= 3 vertices");
    }

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // same QoS as local_costmap's /costmap and /costmap/keepout
    rclcpp::QoS qos(10);
    qos.reliable();
    costmap_pub_ = create_publisher<OccupancyGrid>("/costmap", qos);
    if (use_semantic) {
      semantic_.initialize(static_cast<int8_t>(crosswalk_cost), Costmap2D::OCCUPIED);
      // map_provider latches the grid: transient_local + reliable
      semantic_sub_ = create_subscription<OccupancyGrid>(
        semantic_topic_, rclcpp::QoS(1).transient_local().reliable(),
        [this](OccupancyGrid::ConstSharedPtr m) {semantic_.setGrid(*m);});
      keepout_pub_ = create_publisher<OccupancyGrid>("/costmap/keepout", qos);
    }
    // the traversability node publishes reliable / volatile, depth 1
    grid_sub_ = create_subscription<OccupancyGrid>(
      grid_topic, rclcpp::QoS(1), [this](OccupancyGrid::ConstSharedPtr m) {onGrid(*m);});
    timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] {watchdog();});

    RCLCPP_INFO(get_logger(), "traversability costmap: %s -> /costmap (unknown -> %d)%s",
      grid_topic.c_str(), unknown_cost_, use_semantic ? ", semantic -> /costmap/keepout" : "");
  }

private:
  void onGrid(const OccupancyGrid & g)
  {
    if (g.header.frame_id != odom_frame_ || g.info.width == 0 || g.info.height == 0 ||
      g.data.size() != static_cast<size_t>(g.info.width) * g.info.height)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "dropping grid: frame '%s' (want '%s'), %ux%u, %zu cells",
        g.header.frame_id.c_str(), odom_frame_.c_str(), g.info.width, g.info.height, g.data.size());
      return;
    }
    last_rx_ = now();
    have_grid_ = true;

    fitTo(costmap_, g);
    int8_t * d = costmap_->getData();
    for (size_t k = 0; k < g.data.size(); ++k) {d[k] = g.data[k] < 0 ? unknown_cost_ : g.data[k];}
    clearFootprint(g.header.stamp);

    costmap_msg_.header = g.header;
    costmap_msg_.info = g.info;
    costmap_msg_.data.assign(d, d + g.data.size());
    costmap_pub_->publish(costmap_msg_);

    if (keepout_pub_) {
      fitTo(keepout_, g);
      keepout_->reset(Costmap2D::FREE_SPACE);
      if (!applySemantic(*keepout_)) {
        // no semantic grid or no map->odom: say "unknown", not "free"
        keepout_->reset(Costmap2D::UNKNOWN);
      }
      OccupancyGrid k;
      k.header = g.header;
      k.info = g.info;
      k.data = keepout_->getDataVector();
      keepout_pub_->publish(k);
    }
  }

  // The robot stands on these cells, so whatever the grid remembers there would only
  // trap the controller.
  void clearFootprint(const builtin_interfaces::msg::Time & stamp)
  {
    geometry_msgs::msg::TransformStamped tf;
    try {
      // the traversability node already waited for this transform
      tf = tf_buffer_->lookupTransform(odom_frame_, base_frame_, stamp);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "no %s <- %s at grid time, footprint not cleared: %s",
        odom_frame_.c_str(), base_frame_.c_str(), ex.what());
      return;
    }
    const double x = tf.transform.translation.x, y = tf.transform.translation.y;
    const double yaw = yawOf(tf.transform.rotation), c = std::cos(yaw), s = std::sin(yaw);
    std::vector<std::pair<double, double>> poly;
    for (size_t i = 0; i + 1 < footprint_.size(); i += 2) {
      poly.emplace_back(x + c * footprint_[i] - s * footprint_[i + 1],
                        y + s * footprint_[i] + c * footprint_[i + 1]);
    }
    costmap_->setPolygonCost(poly, Costmap2D::FREE_SPACE);
  }

  bool applySemantic(Costmap2D & target)
  {
    if (!semantic_.isReady()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
        "semantic grid not received yet on %s", semantic_topic_.c_str());
      return false;
    }
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(map_frame_, odom_frame_, tf2::TimePointZero);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "no %s <- %s, keepout unknown: %s", map_frame_.c_str(), odom_frame_.c_str(), ex.what());
      return false;
    }
    semantic_.apply(target, tf.transform.translation.x, tf.transform.translation.y,
                    yawOf(tf.transform.rotation));
    return true;
  }

  // Before the first grid nothing is published: smppi does not run without /costmap.
  void watchdog()
  {
    if (!have_grid_) {return;}
    const double age = (now() - last_rx_).seconds();
    if (age <= grid_timeout_) {return;}
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
      "no traversability grid for %.1f s (timeout %.1f s), publishing UNKNOWN /costmap",
      age, grid_timeout_);
    costmap_msg_.header.stamp = now();
    std::fill(costmap_msg_.data.begin(), costmap_msg_.data.end(), Costmap2D::UNKNOWN);
    costmap_pub_->publish(costmap_msg_);
  }

  std::string odom_frame_, base_frame_, map_frame_, semantic_topic_;
  double grid_timeout_;
  int8_t unknown_cost_;
  std::vector<double> footprint_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<OccupancyGrid>::SharedPtr costmap_pub_, keepout_pub_;
  rclcpp::Subscription<OccupancyGrid>::SharedPtr grid_sub_, semantic_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  SemanticLayer semantic_;
  std::unique_ptr<Costmap2D> costmap_, keepout_;
  OccupancyGrid costmap_msg_;
  bool have_grid_ = false;
  rclcpp::Time last_rx_;
};

}  // namespace local_costmap

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<local_costmap::TraversabilityCostmapNode>());
  rclcpp::shutdown();
  return 0;
}
