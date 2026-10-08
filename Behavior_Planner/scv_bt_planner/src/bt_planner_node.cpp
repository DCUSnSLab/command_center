// scv_bt_planner ROS 노드.
//
// simple_behavior_planner_node.py 의 입출력 계약을 그대로 지키면서 판단만 BT 로 한다.
// 콜백은 원본 로직(순간이동 재정렬, RC 이동 재정렬, 자율 전환 재선택, pause 노드 7/8,
// goal_status 처리)을 이식했고, 틱 루프는 Context → 트리 → 출력 소비 순서다.
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>
#include <command_center_interfaces/msg/controller_goal_status.hpp>
#include <command_center_interfaces/msg/mppi_params.hpp>
#include <command_center_interfaces/msg/multiple_waypoints.hpp>
#include <command_center_interfaces/msg/pause_command.hpp>
#include <command_center_interfaces/msg/planned_path.hpp>
#include <command_center_interfaces/msg/target_waypoints.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <hunter_msgs/msg/hunter_status.hpp>
#include <map_interfaces/msg/utm_layer.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "scv_bt_planner/bt_nodes.hpp"

using namespace std::chrono_literals;
namespace cci = command_center_interfaces::msg;

namespace scv_bt_planner {

class BtPlannerNode : public rclcpp::Node {
public:
  BtPlannerNode() : rclcpp::Node("bt_planner")
  {
    declareParams();
    loadAssets();
    setupIo();
    buildTree();
    const double hz = get_parameter("tick_hz").as_double();
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / hz), [this] { onTick(); });
    RCLCPP_INFO(get_logger(), "[BT] mode=%s prefix='%s' tree=%s zones=%zu/%zu profiles=%zu+%zu",
                mode_.c_str(), prefix_.c_str(), tree_file_.c_str(), zones_.size(), zones_.nodeCount(),
                profiles_.nodeProfileCount(), profiles_.overlayCount());
  }

private:
  // ------------------------------------------------------------ 설정
  void declareParams()
  {
    declare_parameter("mode", "shadow");
    declare_parameter("tick_hz", 20.0);
    declare_parameter("tree_file", "");
    declare_parameter("profiles_file", "");
    declare_parameter("map_file_path", "");
    declare_parameter("current_position_topic", "/odometry/global");
    declare_parameter("planned_path_topic", "/planned_path_detailed");
    declare_parameter("goal_status_topic", "/goal_status");
    declare_parameter("subgoal_topic", "/sub_goal");
    declare_parameter("multiple_waypoints_topic", "/multiple_waypoints");
    declare_parameter("target_waypoints_topic", "/target_waypoints");
    declare_parameter("mppi_params_topic", "/mppi_update_params");   // smppi params_update_sub 와 동일
    declare_parameter("emergency_stop_topic", "/emergency_stop");
    declare_parameter("stop_flag_topic", "/stop_flag");
    declare_parameter("pause_command_topic", "/pause_command");
    declare_parameter("behavior_status_topic", "/behavior_status");
    declare_parameter("assist_request_topic", "/blocked_assist_request");
    declare_parameter("datum_topic", "/map_provider_node/utm");
    declare_parameter("hunter_status_topic", "/hunter_status");
    declare_parameter("join_costmap_topic", "/costmap");
    declare_parameter("localization_hint_topic", "/behavior/localization_hint");
    declare_parameter("waypoint_mode", "multiple");
    declare_parameter("pause_trigger_distance", 0.8);
    declare_parameter("goal_distance_via", 1.6);
    declare_parameter("goal_distance_final", 0.4);
    declare_parameter("join_nearest", false);
    declare_parameter("join_max_approach_m", 40.0);
    declare_parameter("join_check_approach", true);
    declare_parameter("join_clear_half_width_m", 0.45);
    declare_parameter("join_clear_lethal", 90);
    // 위험 지대 경유점 재배치(AvoidHazardWaypoints 노드가 쓴다). 코스트맵 값 >= hazard.lethal 을 위험으로 본다.
    // 50 = SMPPI obstacle_cost_threshold 0.5 와 같은 기준. 구덩이는 코스트맵에서 100 이 되지 않는다 —
    // 가까운 가장자리만 ~61 이고 내부는 0(10/02 챔버 실측, 라이다가 구덩이 안을 못 본다).
    // 기본 비활성(opt-in): 챔버 국소 코스트맵에 위험물 없는 경로 노드 주변에도 지속적인 위험 셀이 있어
    // 정상 노드를 건너뛴다(10/02 scen10-11: H00~H03). 인지·코스트맵 쪽 표현이 정리되기 전에는 시나리오 시험에서만 켠다.
    declare_parameter("hazard.enable", false);
    declare_parameter("hazard.lethal", 50);
    // 위험 셀 뒤쪽(진행 방향) 그림자 길이. 구덩이는 가까운 가장자리만 보이고 안쪽은 라이다 그림자라 코스트 0 —
    // 보이는 가장자리 뒤 이 길이까지 위험으로 본다(10/02 scen6: 그림자 없이 1.0 m 비켜 놓은 경유점이 구덩이를 스쳐 추락).
    declare_parameter("hazard.shadow_m", 1.2);
    declare_parameter("hazard.reach_m", 0.6);   // 재배치 경유점이 목표일 때 도달 임계 [m]
    declare_parameter("hazard.slow_v", 0.5);    // 위험물 감지 중 최고 속도 [m/s]
    declare_parameter("join_pass_radius_m", 2.0);
    declare_parameter("realign_on_engage", true);
    declare_parameter("realign_manual_move_m", 2.0);
    declare_parameter("blocked.detect_sec", 4.0);
    declare_parameter("blocked.progress_eps", 0.15);
    declare_parameter("blocked.wait_timeout", 12.0);
    declare_parameter("blocked.creep_timeout", 10.0);
    declare_parameter("blocked.creep_speed", 0.3);
    declare_parameter("blocked.assist_repeat_sec", 15.0);
    declare_parameter("blocked.near_goal_hold_off", 0.8);
    declare_parameter("probe.enabled", false);
    declare_parameter("probe.max_dist_m", 2.0);
    declare_parameter("turn.min_deg", 35.0);
    declare_parameter("turn.window_m", 6.0);
    declare_parameter("reverse.allow_behind_goal", false);
    declare_parameter("groot2_port", 0);

    mode_ = get_parameter("mode").as_string();
    prefix_ = (mode_ == "active") ? "" : "/bt";
    waypoint_mode_ = get_parameter("waypoint_mode").as_string();
    ctx_.pause_trigger_distance = get_parameter("pause_trigger_distance").as_double();
    ctx_.near_goal_hold_off = get_parameter("blocked.near_goal_hold_off").as_double();
    ctx_.allow_behind_reverse = get_parameter("reverse.allow_behind_goal").as_bool();
    BlockedWaitMonitor::Params bp;
    bp.blocked_detect_sec = get_parameter("blocked.detect_sec").as_double();
    bp.progress_eps = get_parameter("blocked.progress_eps").as_double();
    bp.wait_timeout = get_parameter("blocked.wait_timeout").as_double();
    bp.creep_timeout = get_parameter("blocked.creep_timeout").as_double();
    bp.creep_speed = get_parameter("blocked.creep_speed").as_double();
    bp.assist_repeat_sec = get_parameter("blocked.assist_repeat_sec").as_double();
    bp.probe_enabled = get_parameter("probe.enabled").as_bool();
    bp.probe_max_dist = get_parameter("probe.max_dist_m").as_double();
    ctx_.blocked = BlockedWaitMonitor(bp);
    if (bp.probe_enabled) {
      RCLCPP_WARN(get_logger(), "probe.enabled=true 이지만 far_wall 분류는 아직 미이식 — probe 는 발동하지 않는다");
    }
    goal_via_ = get_parameter("goal_distance_via").as_double();
    goal_final_ = get_parameter("goal_distance_final").as_double();
    join_nearest_ = get_parameter("join_nearest").as_bool();
    join_max_approach_ = get_parameter("join_max_approach_m").as_double();
    join_check_approach_ = get_parameter("join_check_approach").as_bool();
    join_clear_half_w_ = get_parameter("join_clear_half_width_m").as_double();
    join_clear_lethal_ = get_parameter("join_clear_lethal").as_int();
    hazard_enable_ = get_parameter("hazard.enable").as_bool();
    hazard_lethal_ = get_parameter("hazard.lethal").as_int();
    if (hazard_enable_) {
      hazard_shadow_m_ = get_parameter("hazard.shadow_m").as_double();
      hazard_reach_m_ = get_parameter("hazard.reach_m").as_double();
      hazard_slow_v_ = get_parameter("hazard.slow_v").as_double();
      ctx_.hazard_blocked = [this](double ux, double uy) { return shadowBlocked(ux, uy); };
      ctx_.hazard_segment_clear = [this](double x0, double y0, double x1, double y1) {
        return snapSegmentClear(x0, y0, x1, y1);
      };
    }
    join_pass_radius_ = get_parameter("join_pass_radius_m").as_double();
    realign_on_engage_ = get_parameter("realign_on_engage").as_bool();
    realign_move_m_ = get_parameter("realign_manual_move_m").as_double();
  }

  static std::string resolvePackageUri(const std::string& p)
  {
    const std::string pfx = "package://";
    if (p.rfind(pfx, 0) != 0) return p;
    const std::string rest = p.substr(pfx.size());
    const auto slash = rest.find('/');
    if (slash == std::string::npos) return p;
    try {
      return ament_index_cpp::get_package_share_directory(rest.substr(0, slash)) + "/" + rest.substr(slash + 1);
    } catch (...) {
      return p;
    }
  }

  void loadAssets()
  {
    const std::string share = ament_index_cpp::get_package_share_directory("scv_bt_planner");
    tree_file_ = get_parameter("tree_file").as_string();
    if (tree_file_.empty()) tree_file_ = share + "/behavior_trees/scv_behavior.xml";
    std::string profiles_file = get_parameter("profiles_file").as_string();
    if (profiles_file.empty()) profiles_file = share + "/config/behavior_profiles.yaml";
    std::string err;
    if (!profiles_.loadProfiles(profiles_file, &err)) {
      RCLCPP_ERROR(get_logger(), "profiles load failed: %s", err.c_str());
    }
    const std::string smppi = resolvePackageUri(profiles_.smppiConfigPath());
    if (!profiles_.loadBaseline(smppi, &err)) {
      RCLCPP_WARN(get_logger(), "smppi baseline load failed (%s) — default baseline 사용", err.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "smppi baseline: %s", smppi.c_str());
    }
    const std::string map_file = get_parameter("map_file_path").as_string();
    if (!map_file.empty()) {
      if (!zones_.loadFromFile(map_file, &err)) {
        RCLCPP_WARN(get_logger(), "zone table load failed (%s) — 전 노드 sidewalk", err.c_str());
      } else if (zones_.size() == 0) {
        RCLCPP_INFO(get_logger(), "graph.json 에 Zone 속성 없음 — 전 노드 sidewalk (동등성 모드)");
      }
    }
    ctx_.zones = &zones_;
    ctx_.profiles = &profiles_;
  }

  // ------------------------------------------------------------ I/O
  template <typename T>
  typename rclcpp::Publisher<T>::SharedPtr pub(const std::string& param, int depth = 10)
  {
    return create_publisher<T>(prefix_ + get_parameter(param).as_string(), depth);
  }

  void setupIo()
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 출력 (shadow 면 /bt 접두)
    subgoal_pub_ = pub<geometry_msgs::msg::PoseStamped>("subgoal_topic");
    multi_pub_ = pub<cci::MultipleWaypoints>("multiple_waypoints_topic");
    target_pub_ = create_publisher<cci::TargetWaypoints>(
      prefix_ + get_parameter("target_waypoints_topic").as_string(),
      rclcpp::QoS(1).transient_local());
    mppi_pub_ = pub<cci::MPPIParams>("mppi_params_topic");
    estop_pub_ = pub<std_msgs::msg::Bool>("emergency_stop_topic");
    pause_pub_ = pub<cci::PauseCommand>("pause_command_topic");
    status_pub_ = pub<std_msgs::msg::String>("behavior_status_topic");
    assist_pub_ = pub<std_msgs::msg::String>("assist_request_topic");
    hint_pub_ = pub<std_msgs::msg::String>("localization_hint_topic");
    // A/B 비교 전용 (모드 무관하게 /bt/behavior)
    behavior_pub_ = create_publisher<std_msgs::msg::String>("/bt/behavior", 10);

    // 입력
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("current_position_topic").as_string(), 10,
      [this](nav_msgs::msg::Odometry::ConstSharedPtr m) { onOdom(*m); });
    path_sub_ = create_subscription<cci::PlannedPath>(
      get_parameter("planned_path_topic").as_string(), 10,
      [this](cci::PlannedPath::ConstSharedPtr m) { onPath(*m); });
    goal_sub_ = create_subscription<cci::ControllerGoalStatus>(
      get_parameter("goal_status_topic").as_string(), 10,
      [this](cci::ControllerGoalStatus::ConstSharedPtr m) { onGoalStatus(*m); });
    stop_sub_ = create_subscription<std_msgs::msg::Bool>(
      get_parameter("stop_flag_topic").as_string(), 10,
      [this](std_msgs::msg::Bool::ConstSharedPtr m) { ctx_.stop_flag = m->data; });
    datum_sub_ = create_subscription<map_interfaces::msg::UtmLayer>(
      get_parameter("datum_topic").as_string(), rclcpp::QoS(1).transient_local().reliable(),
      [this](map_interfaces::msg::UtmLayer::ConstSharedPtr m) {
        origin_e_ = m->origin_easting; origin_n_ = m->origin_northing;
      });
    hunter_sub_ = create_subscription<hunter_msgs::msg::HunterStatus>(
      get_parameter("hunter_status_topic").as_string(), 10,
      [this](hunter_msgs::msg::HunterStatus::ConstSharedPtr m) { onHunterStatus(*m); });
    if (join_check_approach_ || hazard_enable_) {
      costmap_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        get_parameter("join_costmap_topic").as_string(), 1,
        [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) { costmap_ = m; });
    }
  }

  void buildTree()
  {
    registerScvNodes(factory_, ctx_);
    tree_ = factory_.createTreeFromFile(tree_file_);
    const int port = get_parameter("groot2_port").as_int();
    if (port > 0) {
      groot_ = std::make_unique<BT::Groot2Publisher>(tree_, static_cast<unsigned>(port));
      RCLCPP_INFO(get_logger(), "Groot2 publisher on port %d", port);
    }
  }

  // ------------------------------------------------------------ 콜백 (원본 이식)
  static double yawOf(const geometry_msgs::msg::Quaternion& q) { return tf2::getYaw(q); }

  void onOdom(const nav_msgs::msg::Odometry& m)
  {
    Pose2D p{m.pose.pose.position.x, m.pose.pose.position.y, yawOf(m.pose.pose.orientation)};
    const std::optional<Pose2D> prev = ctx_.pose;
    const double t = rclcpp::Time(m.header.stamp).seconds();
    ctx_.pose = p;
    if (origin_e_) ctx_.pose_utm = std::make_pair(p.x + *origin_e_, p.y + *origin_n_);

    if (ctx_.path.hasPath()) {
      // 통과 이력은 노드 좌표(절대 UTM)와 같은 프레임이어야 한다 → map 좌표를 UTM 으로 올린다
      if (origin_e_) ctx_.path.notePosition(p.x + *origin_e_, p.y + *origin_n_, join_pass_radius_);
    }
    if (prev && unpinned_route_) {
      const double dt = t - prev_pose_t_;
      const double jump = std::hypot(p.x - prev->x, p.y - prev->y);
      if (jump > std::max(2.0, 5.0 * std::max(dt, 0.0))) {
        RCLCPP_WARN(get_logger(), "pose teleport %.1f m (dt=%.2fs) — re-aligning start node", jump, dt);
        ctx_.pending_align = true;
      }
    }
    prev_pose_t_ = t;
    if (ctx_.control_mode == 3 && realign_on_engage_ && unpinned_route_) {
      if (!align_pose_ || std::hypot(p.x - align_pose_->x, p.y - align_pose_->y) > realign_move_m_) {
        ctx_.pending_align = true;
      }
    }
    if (ctx_.pending_align && alignStartToPose()) {
      ctx_.pending_align = false;
      align_pose_ = p;
      ctx_.waypoints_published = false;
    }
  }

  // shadow 전용: 제어기가 보고하는 goal_id 가 내 목표보다 경로상 앞이면 그 노드로 빨리감기한다.
  // shadow 에서는 목표를 내가 아니라 현행 플래너가 주므로, 전환 시점 재합류가 현행보다 0.5~1 s 늦으면
  // (10/01 Pod run2 shadow1: 현행 C02→C03 전진 뒤에 내가 C02 로 재합류) 내 목표 노드의 도달 보고가
  // 영영 오지 않아 C07 완주를 "C02 차단" 으로 오판해 BLOCKED_WAIT→CREEP→ASSIST 를 헛발동한다.
  // active 에서는 목표를 내가 내므로 이 경로가 생기지 않는다 — 결정 로직은 건드리지 않는다.
  void shadowResync(const std::string& goal_id)
  {
    const auto& nodes = ctx_.path.nodes();
    int j = -1;
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (nodes[i].id == goal_id) { j = static_cast<int>(i); break; }
    }
    const int cur = ctx_.path.currentIndex();
    if (j <= cur) return;
    while (ctx_.path.currentIndex() < j) {
      const PathNode* t = ctx_.target();
      if (t) ctx_.path.markGoalCompleted(t->id);
      if (!ctx_.path.advanceToNextNode()) break;
    }
    pause_signal_sent_ = false;
    ctx_.waypoints_published = false;
    ctx_.goal_distance.reset();
    RCLCPP_WARN(get_logger(), "[shadow] target resync %s -> %s (현행 플래너 목표가 앞섬, %d 노드 건너뜀)",
                nodes[cur].id.c_str(), goal_id.c_str(), j - cur);
  }

  bool alignStartToPose()
  {
    if (!ctx_.pose || !ctx_.path.hasPath() || !origin_e_) return false;
    const double ux = ctx_.pose->x + *origin_e_, uy = ctx_.pose->y + *origin_n_;
    const double cap = join_nearest_ ? 0.0 : join_max_approach_;
    PathManager::ApproachClear clear;
    if (join_check_approach_) {
      clear = [this](double x0, double y0, double x1, double y1) { return approachClear(x0, y0, x1, y1); };
    }
    const int idx = ctx_.path.alignToPosition(ux, uy, cap, 1.5, 2, 0, clear, true, join_pass_radius_);
    const PathNode* n = ctx_.path.currentTarget();
    RCLCPP_INFO(get_logger(), "start node unspecified -> join idx %d (%s) by %s%s", idx,
                n ? n->id.c_str() : "?", join_nearest_ ? "nearest" : "min(approach+remaining)",
                clear ? "+approach_clear" : "");
    return true;
  }

  // 틱마다 한 번 코스트맵·TF(map→grid) 스냅샷 — 위험 경유점 재배치가 점/직선 질의를 수백 번 하므로 TF 를 매번 찾지 않는다.
  void takeGridSnap()
  {
    snap_ok_ = false;
    snap_grid_ = costmap_;
    if (!snap_grid_ || !origin_e_) return;
    try {
      const auto tr = tf_buffer_->lookupTransform(snap_grid_->header.frame_id, "map", tf2::TimePointZero);
      snap_tx_ = tr.transform.translation.x; snap_ty_ = tr.transform.translation.y;
      const double th = yawOf(tr.transform.rotation);
      snap_c_ = std::cos(th); snap_s_ = std::sin(th);
      snap_ok_ = true;
    } catch (...) {
    }
  }

  // 절대 UTM 점의 코스트(0-100). 스냅샷 없음·격자 밖·미지는 0(통행 가능, fail-open).
  int snapCost(double ux, double uy) const
  {
    if (!snap_ok_) return 0;
    const double mx = ux - *origin_e_, my = uy - *origin_n_;
    const double gx = snap_c_ * mx - snap_s_ * my + snap_tx_, gy = snap_s_ * mx + snap_c_ * my + snap_ty_;
    const auto& info = snap_grid_->info;
    const int col = static_cast<int>(std::floor((gx - info.origin.position.x) / info.resolution));
    const int row = static_cast<int>(std::floor((gy - info.origin.position.y) / info.resolution));
    if (col < 0 || row < 0 || col >= static_cast<int>(info.width) || row >= static_cast<int>(info.height)) return 0;
    const int v = snap_grid_->data[row * info.width + col];
    return v < 0 ? 0 : v;
  }

  // 점 (ux,uy) 가 위험 셀이거나, 진행 방향(차량 yaw) 뒤쪽 shadow_m 안에 위험 셀이 있으면(= 그 셀의 그림자 안) 위험.
  bool shadowBlocked(double ux, double uy) const
  {
    if (snapCost(ux, uy) >= hazard_lethal_) return true;
    if (hazard_shadow_m_ <= 0.0 || !ctx_.pose) return false;
    const double cx = std::cos(ctx_.pose->yaw), cy = std::sin(ctx_.pose->yaw);
    const double step = snap_ok_ ? std::max<double>(snap_grid_->info.resolution, 0.1) : 0.1;
    for (double s = step; s <= hazard_shadow_m_ + 1e-9; s += step) {
      if (snapCost(ux - s * cx, uy - s * cy) >= hazard_lethal_) return true;
    }
    return false;
  }

  bool snapSegmentClear(double x0, double y0, double x1, double y1) const
  {
    if (!snap_ok_) return true;
    const double d = std::hypot(x1 - x0, y1 - y0);
    if (d <= 1e-6) return true;
    const double nx = -(y1 - y0) / d, ny = (x1 - x0) / d;
    const double step = std::max<double>(snap_grid_->info.resolution, 0.05);
    const int steps = std::max(2, static_cast<int>(d / step));
    for (int k = 0; k <= steps; ++k) {
      const double t = static_cast<double>(k) / steps;
      const double bx = x0 + (x1 - x0) * t, by = y0 + (y1 - y0) * t;
      for (double off : {-join_clear_half_w_, 0.0, join_clear_half_w_}) {
        if (shadowBlocked(bx + nx * off, by + ny * off)) return false;
      }
    }
    return true;
  }

  // (x0,y0)->(x1,y1) 절대 UTM 직선의 코스트맵 통행성. 정보 부족은 통행 가능(fail-open).
  bool approachClear(double x0, double y0, double x1, double y1)
  {
    auto grid = costmap_;
    if (!grid || !origin_e_) return true;
    geometry_msgs::msg::TransformStamped tr;
    try {
      tr = tf_buffer_->lookupTransform(grid->header.frame_id, "map", tf2::TimePointZero);
    } catch (...) {
      return true;
    }
    const double tx = tr.transform.translation.x, ty = tr.transform.translation.y;
    const double th = yawOf(tr.transform.rotation);
    const double c = std::cos(th), s = std::sin(th);
    // UTM → map
    x0 -= *origin_e_; x1 -= *origin_e_; y0 -= *origin_n_; y1 -= *origin_n_;
    const auto res = grid->info.resolution;
    const int W = grid->info.width, H = grid->info.height;
    const double d = std::hypot(x1 - x0, y1 - y0);
    if (d <= 1e-6) return true;
    const double nx = -(y1 - y0) / d, ny = (x1 - x0) / d;
    const int steps = std::max(2, static_cast<int>(d / std::max<double>(res, 0.05)));
    for (int k = 0; k <= steps; ++k) {
      const double t = static_cast<double>(k) / steps;
      const double bx = x0 + (x1 - x0) * t, by = y0 + (y1 - y0) * t;
      for (double off : {-join_clear_half_w_, 0.0, join_clear_half_w_}) {
        const double mx = bx + nx * off, my = by + ny * off;
        const double gx = c * mx - s * my + tx, gy = s * mx + c * my + ty;
        const int col = static_cast<int>((gx - grid->info.origin.position.x) / res);
        const int row = static_cast<int>((gy - grid->info.origin.position.y) / res);
        if (col < 0 || col >= W || row < 0 || row >= H) continue;
        if (grid->data[row * W + col] >= join_clear_lethal_) return false;
      }
    }
    return true;
  }

  void onPath(const cci::PlannedPath& m)
  {
    std::vector<PathNode> nodes;
    for (const auto& n : m.path_data.nodes) {
      nodes.push_back({n.id, n.easting, n.northing, static_cast<int>(n.node_type), n.heading_deg});
    }
    ctx_.path.setPath(std::move(nodes), m.path_id);
    ctx_.hazard.clear();
    completed_logged_ = false;
    ctx_.waypoints_published = false;
    ctx_.pause_sent_for.clear();
    pause_signal_sent_ = false;
    unpinned_route_ = m.start_node_id.empty();
    if (unpinned_route_) {
      if (!alignStartToPose()) ctx_.pending_align = true;
    }
    RCLCPP_INFO(get_logger(), "path %s: %zu nodes (unpinned=%d)", m.path_id.c_str(), ctx_.path.size(),
                unpinned_route_ ? 1 : 0);
  }

  void onGoalStatus(const cci::ControllerGoalStatus& m)
  {
    const PathNode* t = ctx_.target();
    if (t && m.goal_id != t->id && mode_ != "active") shadowResync(m.goal_id);
    t = ctx_.target();
    if (!t || m.goal_id != t->id) {
      if (m.goal_reached && t) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "goal_status ignored: goal_id=%s != target=%s",
                             m.goal_id.c_str(), t->id.c_str());
      }
      return;
    }
    ctx_.goal_distance = m.distance_to_goal;
    // pause 노드 7/8 거리 트리거 (원본 _check_pause_trigger)
    if (!pause_signal_sent_ && m.distance_to_goal <= ctx_.pause_trigger_distance &&
        (t->node_type == 7 || t->node_type == 8)) {
      const double dur = (t->node_type == 7) ? 2.0 : 4.0;
      sendPause(dur, t->id, "Node type " + std::to_string(t->node_type) + " pause");
      pause_signal_sent_ = true;
      ctx_.pause_until = ctx_.now + dur + 1.0;
    }
    if (m.goal_reached && m.status_code == 1) {
      if (!pause_signal_sent_ && (t->node_type == 7 || t->node_type == 8)) {
        sendPause(t->node_type == 7 ? 2.0 : 4.0, t->id,
                  "Node type " + std::to_string(t->node_type) + " pause (on reach)");
      }
      ctx_.path.markGoalCompleted(m.goal_id);
      pause_signal_sent_ = false;
      ctx_.waypoints_published = false;
      ctx_.goal_distance.reset();
      if (ctx_.path.advanceToNextNode()) {
        const PathNode* n = ctx_.target();
        RCLCPP_INFO(get_logger(), "advanced to %s (%d/%zu)", n ? n->id.c_str() : "?", ctx_.path.currentIndex() + 1,
                    ctx_.path.size());
      } else if (!completed_logged_) {
        // 제어기는 최종 목표 도달 후에도 goal_reached=true 를 계속 보낸다 — 한 번만 기록
        RCLCPP_INFO(get_logger(), "Path following completed!");
        completed_logged_ = true;
      }
    } else if (m.status_code == 2) {
      RCLCPP_WARN(get_logger(), "goal %s failed (d=%.2f)", m.goal_id.c_str(), m.distance_to_goal);
      ctx_.waypoints_published = false;
    } else if (m.status_code == 3) {
      RCLCPP_WARN(get_logger(), "goal %s aborted — emergency stop", m.goal_id.c_str());
      ctx_.emergency_stop = true;
    }
  }

  void onHunterStatus(const hunter_msgs::msg::HunterStatus& m)
  {
    const int mode = m.control_mode;
    if (mode != ctx_.control_mode) {
      const char* name = mode == 0 ? "대기" : mode == 1 ? "CAN(자율)" : mode == 3 ? "RC(수동)" : "?";
      RCLCPP_INFO(get_logger(), "[MODE] hunter control_mode -> %d (%s)%s", mode, name,
                  mode == 1 ? "" : " — 차단 에스컬레이션 보류");
      const int prev = ctx_.control_mode;
      ctx_.control_mode = mode;
      if (prev != -1 && mode == 1 && realign_on_engage_ && unpinned_route_) {
        RCLCPP_INFO(get_logger(), "[MODE] 자율 전환 — 현재 위치로 합류 노드 재선택");
        ctx_.pending_align = true;
      }
    }
  }

  // ------------------------------------------------------------ 틱
  static double steadyNow()
  {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void onTick()
  {
    ctx_.now = steadyNow();
    ctx_.resetTick();
    updateAnchorDelta();

    if (hazard_enable_) takeGridSnap();
    tree_.tickOnce();
    for (const auto& s : ctx_.hazard_log) RCLCPP_WARN(get_logger(), "[HAZARD] %s", s.c_str());

    if (ctx_.request_stop) {
      std_msgs::msg::Bool b; b.data = true;
      estop_pub_->publish(b);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Emergency stop published! (%s)", ctx_.stop_reason.c_str());
      publishBehaviorIfChanged();
      return;
    }
    if (!ctx_.path.isFollowing() || !ctx_.pose || !ctx_.path.hasPath()) return;

    // 안전 pause (node_type 9 + stop_flag): 원본 safety_monitor 0.1 s 간격
    if (ctx_.safety_pause_due && ctx_.now - last_safety_pause_t_ >= 0.1) {
      sendPause(2.0, "safety_stop_type_9", "Safety stop: obstacle detected (node_type 9)");
      last_safety_pause_t_ = ctx_.now;
    }
    for (const auto& pr : ctx_.pause_requests) sendPause(pr.duration, pr.node_id, pr.reason);

    // 행동 프로필 → MPPIParams (변경 시에만)
    applyBehavior();

    // 정체 에스컬레이션 액션
    for (const auto& a : ctx_.blocked_actions) handleBlockedAction(a);

    // 웨이포인트
    if (!ctx_.waypoints_published || ctx_.need_publish_waypoints) {
      if (ctx_.need_publish_waypoints) {
        RCLCPP_INFO(get_logger(), "map→odom anchor moved %.2f m / %+.1f deg — refreshing waypoints",
                    ctx_.anchor_moved_m, ctx_.anchor_moved_deg);
      }
      publishWaypoints();
    } else if (ctx_.now - last_target_republish_t_ >= 1.0) {
      publishTargetWaypoints();
      last_target_republish_t_ = ctx_.now;
    }

    if (ctx_.localization_hint && *ctx_.localization_hint != last_hint_) {
      std_msgs::msg::String s; s.data = *ctx_.localization_hint;
      hint_pub_->publish(s);
      last_hint_ = *ctx_.localization_hint;
    }
    publishBehaviorIfChanged();
  }

  void updateAnchorDelta()
  {
    try {
      const auto tr = tf_buffer_->lookupTransform("odom", "map", tf2::TimePointZero);
      const double x = tr.transform.translation.x, y = tr.transform.translation.y;
      const double yaw = yawOf(tr.transform.rotation);
      last_tf_ = Pose2D{x, y, yaw};
      if (tf_at_publish_) {
        ctx_.anchor_moved_m = std::hypot(x - tf_at_publish_->x, y - tf_at_publish_->y);
        ctx_.anchor_moved_deg = std::atan2(std::sin(yaw - tf_at_publish_->yaw), std::cos(yaw - tf_at_publish_->yaw)) * 180.0 / M_PI;
      } else {
        ctx_.anchor_moved_m = 0.0; ctx_.anchor_moved_deg = 0.0;
      }
    } catch (...) {
      ctx_.anchor_moved_m = 0.0; ctx_.anchor_moved_deg = 0.0;
    }
  }

  // ------------------------------------------------------------ 출력
  void applyBehavior()
  {
    const int node_type = ctx_.effectiveNodeType();
    const bool is_final = ctx_.path.isFinalNode();
    // 현재 목표가 위험 지대 재배치 경유점이면 도달 임계를 좁힌다 — 경유 1.6 m 그대로면 1.5 m 옆으로 옮긴 점이
    // 원래 경로 위(구덩이 바로 앞)에서 '도달' 처리돼 재배치가 궤적에 반영되지 않는다(10/02 scen7).
    const PathNode* tgt = ctx_.target();
    const auto toff = tgt ? ctx_.hazard.offset(tgt->id) : std::make_pair(0.0, 0.0);
    const bool shifted = (toff.first != 0.0 || toff.second != 0.0);
    const bool hazard_slow = hazard_enable_ && ctx_.hazardRecent();
    const std::string key = std::to_string(node_type) + "|" + ctx_.overlay + "|" + (is_final ? "F" : "V") +
                            (shifted ? "|S" : "") + (hazard_slow ? "|H" : "");
    if (key == last_behavior_key_ && !creep_active_) return;
    if (creep_active_) return;   // creep 중에는 creep_off 에서 복원
    std::string desc; int code = 0;
    ParamMap p = profiles_.compute(node_type, ctx_.overlay, &desc, &code);
    p["goal_reached_threshold"] = is_final ? goal_final_ : (shifted ? std::min(goal_via_, hazard_reach_m_) : goal_via_);
    // 위험물 감지 중 감속 — 제어기가 재배치 경유점을 따라 꺾을 시간을 번다
    if (hazard_slow) p["max_linear_velocity"] = std::min(p["max_linear_velocity"], hazard_slow_v_);
    std::string why;
    if (!Profiles::validate(p, &why)) {
      RCLCPP_ERROR(get_logger(), "invalid behavior params (%s): %s", key.c_str(), why.c_str());
      return;
    }
    sendMppiParams(p, code, desc);
    RCLCPP_INFO(get_logger(), "[BEHAVIOR] %s type=%d overlay='%s' final=%d max_v=%.2f reach=%.2f%s", desc.c_str(), node_type,
                ctx_.overlay.c_str(), is_final ? 1 : 0, p["max_linear_velocity"], p["goal_reached_threshold"],
                shifted ? " (hazard-shifted target)" : (hazard_slow ? " (hazard ahead)" : ""));
    last_behavior_key_ = key;
    last_params_ = p; last_code_ = code; last_desc_ = desc;
  }

  void sendMppiParams(const ParamMap& p, int code, const std::string& desc)
  {
    auto g = [&](const char* k, double d) { auto it = p.find(k); return it == p.end() ? d : it->second; };
    cci::MPPIParams m;
    m.header.stamp = now();
    m.header.frame_id = "behavior_" + std::to_string(code);
    m.update_vehicle = true;
    m.max_linear_velocity = g("max_linear_velocity", 0.0);
    m.min_linear_velocity = g("min_linear_velocity", 0.0);
    m.max_angular_velocity = g("max_angular_velocity", 1.16);
    m.min_angular_velocity = g("min_angular_velocity", -1.16);
    m.wheelbase = g("wheelbase", 0.65);
    m.max_steering_angle = g("max_steering_angle", 0.3665);
    m.radius = g("radius", 0.6);
    m.footprint_padding = g("footprint_padding", 0.15);
    m.update_costs = true;
    m.goal_weight = g("goal_weight", 30.0);
    m.obstacle_weight = g("obstacle_weight", 100.0);
    m.update_lookahead = true;
    m.lookahead_base_distance = g("lookahead_base_distance", 1.0);
    m.lookahead_velocity_factor = g("lookahead_velocity_factor", 0.4);
    m.lookahead_min_distance = g("lookahead_min_distance", 1.0);
    m.lookahead_max_distance = g("lookahead_max_distance", 6.0);
    m.update_goal_critic = true;
    m.respect_reverse_heading = g("respect_reverse_heading", 0.0) > 0.5;
    m.update_control = true;
    m.goal_reached_threshold = g("goal_reached_threshold", 2.0);
    m.control_frequency = g("control_frequency", 20.0);
    m.force_stop = false;
    m.current_behavior_type = code;
    m.current_behavior_desc = desc;
    mppi_pub_->publish(m);
  }

  void handleBlockedAction(const std::string& a)
  {
    if (a.rfind("announce:", 0) == 0) {
      const std::string state = a.substr(9);
      std_msgs::msg::String s; s.data = state;
      status_pub_->publish(s);
      if (state == "NORMAL") RCLCPP_INFO(get_logger(), "[BLOCKED] cleared -> NORMAL");
      else RCLCPP_WARN(get_logger(), "[BLOCKED] state -> %s", state.c_str());
    } else if (a == "creep_on" && hazard_enable_ && ctx_.hazardRecent()) {
      // 위험물(구덩이·장애물) 앞 정체에서 저속 전진은 위험물 쪽으로 밀어 넣는다(10/02 scen8 추락) — 하지 않는다.
      RCLCPP_WARN(get_logger(), "[BLOCKED] creep suppressed — hazard ahead (wait for assist)");
    } else if (a == "creep_on") {
      ParamMap p = last_params_.empty() ? profiles_.compute(ctx_.effectiveNodeType(), ctx_.overlay) : last_params_;
      p["max_linear_velocity"] = ctx_.blocked.params().creep_speed;
      p["min_linear_velocity"] = 0.0;
      sendMppiParams(p, last_code_, "BLOCKED creep (" + last_desc_ + ")");
      creep_active_ = true;
      RCLCPP_WARN(get_logger(), "[BLOCKED] creep mode ON: max_v=%.2f m/s", ctx_.blocked.params().creep_speed);
    } else if (a == "creep_off") {
      creep_active_ = false;
      last_behavior_key_.clear();   // 다음 applyBehavior 에서 복원 발행
      applyBehavior();
      RCLCPP_INFO(get_logger(), "[BLOCKED] behavior params restored");
    } else if (a == "assist") {
      std_msgs::msg::String s;
      const PathNode* t = ctx_.target();
      s.data = "blocked at node " + std::string(t ? t->id : "?") +
               ": no in-corridor avoidance path; operator attention requested";
      assist_pub_->publish(s);
      RCLCPP_ERROR(get_logger(), "[BLOCKED] assist requested: %s", s.data.c_str());
    } else if (a == "probe_on" || a == "probe_off") {
      RCLCPP_WARN(get_logger(), "[PROBE] %s (미이식 — 무시)", a.c_str());
    }
  }

  void sendPause(double duration, const std::string& node_id, const std::string& reason)
  {
    cci::PauseCommand m;
    m.header.stamp = now();
    m.header.frame_id = "bt_planner";
    m.pause_duration = duration;
    m.node_id = node_id;
    m.reason = reason;
    pause_pub_->publish(m);
    RCLCPP_INFO(get_logger(), "pause %.1fs @%s: %s", duration, node_id.c_str(), reason.c_str());
  }

  std::optional<geometry_msgs::msg::PoseStamped> makeOdomPose(const PathNode& n)
  {
    if (!origin_e_) return std::nullopt;
    geometry_msgs::msg::PoseStamped mp;
    mp.header.stamp = now();
    mp.header.frame_id = "map";
    const auto off = ctx_.hazard.offset(n.id);   // 위험 지대 재배치(없으면 0)
    mp.pose.position.x = n.x + off.first - *origin_e_;
    mp.pose.position.y = n.y + off.second - *origin_n_;
    mp.pose.position.z = 0.0;
    const double yaw = std::fmod(n.heading_deg, 360.0) * M_PI / 180.0;
    mp.pose.orientation.z = std::sin(yaw / 2.0);
    mp.pose.orientation.w = std::cos(yaw / 2.0);
    try {
      const auto tr = tf_buffer_->lookupTransform("odom", "map", tf2::TimePointZero);
      geometry_msgs::msg::PoseStamped op;
      tf2::doTransform(mp, op, tr);
      op.header.frame_id = n.id;   // 원본 규약: frame_id 에 노드 ID
      return op;
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "map->odom TF 미가용, waypoint 스킵: %s", e.what());
      return std::nullopt;
    }
  }

  void publishTargetWaypoints()
  {
    const PathNode* t = ctx_.target();
    if (!t) return;
    cci::TargetWaypoints m;
    m.header.stamp = now();
    m.path_id = ctx_.path.pathId();
    m.current_node_id = t->id;
    for (const auto& n : ctx_.path.nextNodes(3)) m.next_node_ids.push_back(n.id);
    m.current_waypoint_index = ctx_.path.currentIndex();
    m.total_waypoints = static_cast<int>(ctx_.path.size());
    m.is_final_waypoint = ctx_.path.isFinalNode();
    m.speed_limit = 0.0;
    m.goal_reached_threshold = 0.0;
    m.recalc_mode = "";
    target_pub_->publish(m);
  }

  void publishWaypoints()
  {
    const PathNode* t = ctx_.target();
    if (!t) return;
    publishTargetWaypoints();
    if (waypoint_mode_ != "external") {
      if (!origin_e_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "datum(/map_provider_node/utm) 미수신 — waypoint 발행 대기");
        return;
      }
      auto cur = makeOdomPose(*t);
      if (!cur) return;
      if (waypoint_mode_ == "single" || waypoint_mode_ == "both") subgoal_pub_->publish(*cur);
      if (waypoint_mode_ != "single") {
        cci::MultipleWaypoints m;
        m.header.stamp = now();
        m.header.frame_id = "odom";
        m.current_goal = *cur;
        m.current_goal_node_type = t->node_type;
        m.current_goal_reverse_heading = (t->node_type == 2 || t->node_type == 4);
        for (const auto& n : ctx_.path.nextNodes(3)) {
          auto wp = makeOdomPose(n);
          if (!wp) continue;
          m.next_waypoints.push_back(*wp);
          m.next_waypoints_node_types.push_back(n.node_type);
          m.next_waypoints_reverse_heading.push_back(n.node_type == 2 || n.node_type == 4);
        }
        m.path_id = ctx_.path.pathId();
        m.current_waypoint_index = ctx_.path.currentIndex();
        m.total_waypoints = static_cast<int>(ctx_.path.size());
        m.is_final_waypoint = ctx_.path.isFinalNode();
        multi_pub_->publish(m);
      }
    }
    ctx_.waypoints_published = true;
    tf_at_publish_ = last_tf_;
    last_target_republish_t_ = ctx_.now;
  }

  void publishBehaviorIfChanged()
  {
    const PathNode* t = ctx_.target();
    const std::string s = ctx_.behavior_name + "|" + ctx_.blocked.state() + "|" + (t ? t->id : "-") +
                          "|zone=" + ctx_.zone() + "|type=" + std::to_string(ctx_.effectiveNodeType());
    if (s == last_behavior_pub_) return;
    std_msgs::msg::String m; m.data = s;
    behavior_pub_->publish(m);
    last_behavior_pub_ = s;
  }

  // ------------------------------------------------------------ 멤버
  Context ctx_;
  ZoneTable zones_;
  Profiles profiles_;
  BT::BehaviorTreeFactory factory_;
  BT::Tree tree_;
  std::unique_ptr<BT::Groot2Publisher> groot_;
  std::string mode_, prefix_, waypoint_mode_, tree_file_;
  double goal_via_ = 1.6, goal_final_ = 0.4;
  bool join_nearest_ = false, join_check_approach_ = true, realign_on_engage_ = true;
  double join_max_approach_ = 40.0, join_clear_half_w_ = 0.45, join_pass_radius_ = 2.0, realign_move_m_ = 2.0;
  int join_clear_lethal_ = 90;
  bool hazard_enable_ = false;
  int hazard_lethal_ = 50;
  double hazard_shadow_m_ = 1.2;
  double hazard_reach_m_ = 0.6;
  double hazard_slow_v_ = 0.5;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr snap_grid_;
  bool snap_ok_ = false;
  double snap_tx_ = 0.0, snap_ty_ = 0.0, snap_c_ = 1.0, snap_s_ = 0.0;
  std::optional<double> origin_e_, origin_n_;
  bool unpinned_route_ = false, pause_signal_sent_ = false, creep_active_ = false, completed_logged_ = false;
  std::optional<Pose2D> align_pose_, last_tf_, tf_at_publish_;
  double prev_pose_t_ = 0.0, last_safety_pause_t_ = 0.0, last_target_republish_t_ = 0.0;
  std::string last_behavior_key_, last_behavior_pub_, last_hint_, last_desc_;
  ParamMap last_params_;
  int last_code_ = 1;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr costmap_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr subgoal_pub_;
  rclcpp::Publisher<cci::MultipleWaypoints>::SharedPtr multi_pub_;
  rclcpp::Publisher<cci::TargetWaypoints>::SharedPtr target_pub_;
  rclcpp::Publisher<cci::MPPIParams>::SharedPtr mppi_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr estop_pub_;
  rclcpp::Publisher<cci::PauseCommand>::SharedPtr pause_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_, assist_pub_, hint_pub_, behavior_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<cci::PlannedPath>::SharedPtr path_sub_;
  rclcpp::Subscription<cci::ControllerGoalStatus>::SharedPtr goal_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr stop_sub_;
  rclcpp::Subscription<map_interfaces::msg::UtmLayer>::SharedPtr datum_sub_;
  rclcpp::Subscription<hunter_msgs::msg::HunterStatus>::SharedPtr hunter_sub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
};

}  // namespace scv_bt_planner

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<scv_bt_planner::BtPlannerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
