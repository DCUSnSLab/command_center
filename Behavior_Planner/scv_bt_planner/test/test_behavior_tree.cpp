// 트리 수준 결정 시험 — 실제 XML(behavior_trees/scv_behavior.xml) 을 로드해 Context 입력 → 틱 → 출력을 확인한다.
// ROS 없이 돈다. 노드 단위 시험(path_manager·blocked_wait·profiles)이 못 보는 것: 노드 순서(안전 가드 → 임무 준비 →
// 회전 > 구역 > 보도 기본), 포트 값(min_deg 35 / window_m 6 / crosswalk pause 2 s), 데코레이터 결합.
// 마지막 시험은 틱 비용(µs·heap 할당 수) 벤치마크 — 최적화 회귀 지표.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>

#include "scv_bt_planner/bt_nodes.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif
#ifndef BT_TREE_DIR
#define BT_TREE_DIR "."
#endif

// ---- heap 할당 계수기(벤치마크용) -------------------------------------------------------------
static std::atomic<size_t> g_allocs{0};
void* operator new(std::size_t n)
{
  ++g_allocs;
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

using namespace scv_bt_planner;

namespace {

PathNode node(const std::string& id, double x, double y, int type = 1, double heading = 0.0)
{
  PathNode n; n.id = id; n.x = x; n.y = y; n.node_type = type; n.heading_deg = heading;
  return n;
}

// 동쪽으로 5 m 간격 직선
std::vector<PathNode> straight(int n, int type = 1)
{
  std::vector<PathNode> v;
  for (int i = 0; i < n; ++i) v.push_back(node("S" + std::to_string(i), 5.0 * i, 0.0, type));
  return v;
}

// 두 번째 노드에서 꺾임: sign=-1 우회전(남쪽), +1 좌회전(북쇽), deg 는 꺾임 각
std::vector<PathNode> bend(double deg, int sign, int type_at_turn = 1)
{
  const double r = deg * M_PI / 180.0;
  std::vector<PathNode> v{node("B0", 0.0, 0.0), node("B1", 5.0, 0.0, type_at_turn)};
  const double dx = 5.0 * std::cos(r), dy = sign * 5.0 * std::sin(r);
  v.push_back(node("B2", 5.0 + dx, dy));
  v.push_back(node("B3", 5.0 + 2 * dx, 2 * dy));
  return v;
}

bool has(const std::vector<std::string>& v, const std::string& s)
{
  for (const auto& x : v) if (x == s) return true;
  return false;
}

struct Fixture {
  Context ctx;
  ZoneTable zones;
  Profiles profiles;
  BT::BehaviorTreeFactory factory;
  BT::Tree tree;

  Fixture()
  {
    ctx.zones = &zones;
    ctx.profiles = &profiles;
    registerScvNodes(factory, ctx);
    tree = factory.createTreeFromFile(std::string(BT_TREE_DIR) + "/scv_behavior.xml");
  }
  // 임무 중 상태: 경로·위치·자율 모드·목표 거리
  void mission(std::vector<PathNode> nodes, int target_index = 1, double goal_distance = 3.0)
  {
    ctx.path.setPath(std::move(nodes), "p1");
    for (int i = 0; i < target_index; ++i) ctx.path.advanceToNextNode();
    ctx.pose = Pose2D{0.0, 0.0, 0.0};
    ctx.control_mode = 1;
    ctx.goal_distance = goal_distance;
  }
  BT::NodeStatus tick(double now = 100.0)
  {
    ctx.now = now;
    ctx.resetTick();
    return tree.tickOnce();
  }
};

}  // namespace

TEST(Tree, NoPathRootFails)
{
  Fixture f;
  f.ctx.pose = Pose2D{};
  EXPECT_EQ(f.tick(), BT::NodeStatus::FAILURE);
  EXPECT_TRUE(f.ctx.behavior_name.empty());
  EXPECT_FALSE(f.ctx.request_stop);
  EXPECT_TRUE(f.ctx.blocked_actions.empty());
}

TEST(Tree, PendingAlignBlocksMission)
{
  Fixture f;
  f.mission(straight(4));
  f.ctx.pending_align = true;
  EXPECT_EQ(f.tick(), BT::NodeStatus::FAILURE);
  EXPECT_TRUE(f.ctx.behavior_name.empty());
  f.ctx.pending_align = false;
  EXPECT_EQ(f.tick(), BT::NodeStatus::SUCCESS);
}

TEST(Tree, EmergencyStopEndsTick)
{
  Fixture f;
  f.mission(straight(4));
  f.ctx.emergency_stop = true;
  EXPECT_EQ(f.tick(), BT::NodeStatus::FAILURE);
  EXPECT_TRUE(f.ctx.request_stop);
  EXPECT_EQ(f.ctx.stop_reason, "emergency_stop");
  EXPECT_EQ(f.ctx.behavior_name, "STOP:emergency_stop");
  EXPECT_TRUE(f.ctx.overlay.empty()) << "Drive 까지 가지 않는다";
  EXPECT_FALSE(f.ctx.shouldBeMoving());
}

TEST(Tree, StraightIsSidewalkEquivalence)
{
  Fixture f;
  f.mission(straight(4));
  EXPECT_EQ(f.tick(), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(f.ctx.behavior_name, "sidewalk");
  EXPECT_TRUE(f.ctx.overlay.empty()) << "sidewalk = node_type 프로필만 (simple 과 동일)";
  ASSERT_TRUE(f.ctx.localization_hint.has_value());
  EXPECT_EQ(*f.ctx.localization_hint, "gps");
  EXPECT_EQ(f.ctx.zone(), "sidewalk");
  EXPECT_EQ(f.ctx.effectiveNodeType(), 1);
  EXPECT_TRUE(f.ctx.pause_requests.empty());
  EXPECT_FALSE(f.ctx.need_publish_waypoints);
}

TEST(Tree, TurnDetectionUsesMinDegAndWindow)
{
  {
    Fixture f;
    f.mission(bend(90.0, -1), 1, 3.0);
    f.tick();
    EXPECT_EQ(f.ctx.behavior_name, "turn_right");
    EXPECT_EQ(f.ctx.overlay, "turn_right");
    EXPECT_NEAR(f.ctx.turnDegAtTarget(), -90.0, 1e-6);
  }
  {
    Fixture f;
    f.mission(bend(90.0, +1), 1, 3.0);
    f.tick();
    EXPECT_EQ(f.ctx.behavior_name, "turn_left");
    EXPECT_NEAR(f.ctx.turnDegAtTarget(), 90.0, 1e-6);
  }
  {
    Fixture f;   // 창(6 m) 밖이면 아직 보도 주행
    f.mission(bend(90.0, -1), 1, 7.0);
    f.tick();
    EXPECT_EQ(f.ctx.behavior_name, "sidewalk");
  }
  {
    Fixture f;   // 35° 미만은 회전이 아니다
    f.mission(bend(30.0, -1), 1, 3.0);
    f.tick();
    EXPECT_EQ(f.ctx.behavior_name, "sidewalk");
  }
  {
    Fixture f;   // 목표 거리 미상(goal_status 전) → 창 판정 불가 → 회전 아님
    f.mission(bend(90.0, -1), 1, 3.0);
    f.ctx.goal_distance.reset();
    f.tick();
    EXPECT_EQ(f.ctx.behavior_name, "sidewalk");
  }
}

TEST(Tree, FirstNodeTurnUsesHeading)
{
  Fixture f;
  // 첫 노드(진입 방향 = heading 0 동쪽), 다음 노드가 북쪽 → +90 좌회전
  f.mission({node("F0", 0.0, 0.0, 1, 0.0), node("F1", 0.0, 5.0), node("F2", 0.0, 10.0)}, 0, 3.0);
  f.tick();
  EXPECT_NEAR(f.ctx.turnDegAtTarget(), 90.0, 1e-6);
  EXPECT_EQ(f.ctx.behavior_name, "turn_left");
  // heading 90 이면 직진
  Fixture g;
  g.mission({node("F0", 0.0, 0.0, 1, 90.0), node("F1", 0.0, 5.0), node("F2", 0.0, 10.0)}, 0, 3.0);
  g.tick();
  EXPECT_NEAR(g.ctx.turnDegAtTarget(), 0.0, 1e-6);
  EXPECT_EQ(g.ctx.behavior_name, "sidewalk");
  // 마지막 노드: 다음이 없으니 0
  Fixture h;
  h.mission(bend(90.0, -1), 3, 1.0);
  h.tick();
  EXPECT_DOUBLE_EQ(h.ctx.turnDegAtTarget(), 0.0);
  EXPECT_EQ(h.ctx.behavior_name, "sidewalk");
}

TEST(Tree, ReverseNodeTypeBeatsTurn)
{
  Fixture f;
  f.mission(bend(90.0, -1, /*type_at_turn=*/2), 1, 3.0);
  f.tick();
  EXPECT_EQ(f.ctx.behavior_name, "reverse");
  EXPECT_TRUE(f.ctx.overlay.empty()) << "reverse 는 node_type 2 프로필 그대로";
  EXPECT_EQ(f.ctx.effectiveNodeType(), 2);
  Fixture g;
  g.mission(straight(4, 4), 1, 3.0);
  g.tick();
  EXPECT_EQ(g.ctx.behavior_name, "reverse");
}

TEST(Tree, ZoneOverlaysAndCrosswalkPause)
{
  // test/data/graph_zones.json: N0002 road, N0003 crosswalk, N0004 gps_denied (N0001 미지정 → sidewalk)
  const std::vector<PathNode> zoned{node("N0001", 0, 0), node("N0002", 5, 0), node("N0003", 10, 0, 3),
                                    node("N0004", 10, 5, 1, 90.0)};
  {
    Fixture f;
    ASSERT_TRUE(f.zones.loadFromFile(std::string(TEST_DATA_DIR) + "/graph_zones.json"));
    f.mission(zoned, 1, 3.0);
    f.tick();
    EXPECT_EQ(f.ctx.zone(), "road");
    EXPECT_EQ(f.ctx.behavior_name, "road");
    EXPECT_EQ(f.ctx.overlay, "road");
    EXPECT_EQ(*f.ctx.localization_hint, "gps");
    EXPECT_TRUE(f.ctx.pause_requests.empty());
  }
  {
    // 횡단보도 노드가 목표이고 그 노드에서 좌회전(90°) → 행동은 turn_left 가 이기지만, 진입 전 정지는 그대로 발동(ForceSuccess 분기)
    Fixture f;
    ASSERT_TRUE(f.zones.loadFromFile(std::string(TEST_DATA_DIR) + "/graph_zones.json"));
    f.mission(zoned, 2, 3.0);
    f.tick(100.0);
    EXPECT_EQ(f.ctx.behavior_name, "turn_left");
    EXPECT_TRUE(f.ctx.pause_requests.empty()) << "pause_trigger_distance(0.8) 밖";
    f.ctx.goal_distance = 0.5;
    f.tick(101.0);
    ASSERT_EQ(f.ctx.pause_requests.size(), 1u);
    EXPECT_DOUBLE_EQ(f.ctx.pause_requests[0].duration, 2.0);
    EXPECT_EQ(f.ctx.pause_requests[0].node_id, "N0003");
    EXPECT_DOUBLE_EQ(f.ctx.pause_until, 101.0 + 2.0 + 1.0);
    EXPECT_FALSE(f.ctx.shouldBeMoving()) << "pause 중엔 정체로 보지 않는다";
    f.tick(101.05);
    EXPECT_TRUE(f.ctx.pause_requests.empty()) << "노드당 한 번만";
    // 회전 창 밖(먼 거리)이면 crosswalk 오버레이
    f.ctx.goal_distance = 7.0;
    f.tick(102.0);
    EXPECT_EQ(f.ctx.behavior_name, "crosswalk");
    EXPECT_EQ(f.ctx.overlay, "crosswalk");
  }
  {
    Fixture f;
    ASSERT_TRUE(f.zones.loadFromFile(std::string(TEST_DATA_DIR) + "/graph_zones.json"));
    f.mission(zoned, 3, 3.0);
    f.tick();
    EXPECT_EQ(f.ctx.behavior_name, "gps_denied");
    EXPECT_EQ(*f.ctx.localization_hint, "pcd_hold");
    EXPECT_EQ(f.ctx.last_selected_zone, "gps_denied");
  }
}

TEST(Tree, SafetyPauseOnlyForType9WithStopFlag)
{
  Fixture f;
  f.mission(straight(4, 9));
  f.tick();
  EXPECT_FALSE(f.ctx.safety_pause_due);
  EXPECT_TRUE(f.ctx.shouldBeMoving());
  f.ctx.stop_flag = true;
  f.tick();
  EXPECT_TRUE(f.ctx.safety_pause_due);
  EXPECT_FALSE(f.ctx.shouldBeMoving());
  EXPECT_EQ(f.ctx.behavior_name, "sidewalk") << "안전 pause 는 행동 선택을 막지 않는다(항상 SUCCESS)";
  Fixture g;
  g.mission(straight(4, 1));
  g.ctx.stop_flag = true;
  g.tick();
  EXPECT_FALSE(g.ctx.safety_pause_due) << "type 9 가 아니면 stop_flag 무시";
}

TEST(Tree, RefreshWaypointsThresholds)
{
  Fixture f;
  f.mission(straight(4));
  f.ctx.waypoints_published = true;
  f.ctx.anchor_moved_m = 0.4; f.ctx.anchor_moved_deg = 4.0;
  f.tick();
  EXPECT_FALSE(f.ctx.need_publish_waypoints);
  f.ctx.anchor_moved_m = 0.6;
  f.tick();
  EXPECT_TRUE(f.ctx.need_publish_waypoints);
  f.ctx.anchor_moved_m = 0.0; f.ctx.anchor_moved_deg = -6.0;
  f.tick();
  EXPECT_TRUE(f.ctx.need_publish_waypoints) << "회전은 절대값";
  f.ctx.waypoints_published = false;   // 아직 발행 전이면 요청하지 않는다(노드가 어차피 발행)
  f.tick();
  EXPECT_FALSE(f.ctx.need_publish_waypoints);
}

TEST(Tree, BlockedEscalationThroughTree)
{
  Fixture f;
  f.mission(straight(4));
  bool announced = false;
  for (double t = 0.0; t < 4.5; t += 0.05) {
    f.tick(t);
    if (has(f.ctx.blocked_actions, "announce:BLOCKED_WAIT")) {
      announced = true;
      EXPECT_GE(t, 4.0 - 1e-9) << "detect_sec 4 s 전엔 발동하지 않는다";
    }
  }
  EXPECT_TRUE(announced);
  EXPECT_EQ(f.ctx.blocked.state(), "BLOCKED_WAIT");
  EXPECT_EQ(f.ctx.behavior_name, "sidewalk") << "에스컬레이션은 행동 선택 위에 얹힌다";
  // 진행하면 복귀
  f.ctx.pose = Pose2D{1.0, 0.0, 0.0};
  f.tick(5.0);
  EXPECT_TRUE(has(f.ctx.blocked_actions, "announce:NORMAL"));
}

TEST(Tree, BlockedHeldWhileNotListening)
{
  Fixture f;
  f.mission(straight(4));
  f.ctx.control_mode = 3;   // RC 수동
  for (double t = 0.0; t < 20.0; t += 0.05) f.tick(t);
  EXPECT_EQ(f.ctx.blocked.state(), "NORMAL");
  // 목표 근접(near_goal_hold_off 0.8) 도 정체 아님
  Fixture g;
  g.mission(straight(4), 1, 0.5);
  for (double t = 0.0; t < 20.0; t += 0.05) g.tick(t);
  EXPECT_EQ(g.ctx.blocked.state(), "NORMAL");
}

// 틱 비용 벤치마크: 1,000 s 분량(20 Hz × 20,000틱)의 보도 직진 + 회전 혼합. 결과는 ctest 속성과 stdout 으로.
TEST(Tree, TickCostBenchmark)
{
  Fixture f;
  ASSERT_TRUE(f.zones.loadFromFile(std::string(TEST_DATA_DIR) + "/graph_zones.json"));
  f.mission(bend(90.0, -1), 1, 3.0);
  f.ctx.waypoints_published = true;
  constexpr int N = 20000;
  for (int i = 0; i < 200; ++i) f.tick(i * 0.05);   // 워밍업
  const size_t a0 = g_allocs.load();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < N; ++i) {
    f.ctx.pose = Pose2D{0.01 * i, 0.0, 0.0};   // 진행 중 (정체 아님)
    f.tick(10.0 + i * 0.05);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const size_t a1 = g_allocs.load();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
  const double allocs = static_cast<double>(a1 - a0) / N;
  std::printf("[BENCH] tick avg %.2f us, heap allocs/tick %.1f (N=%d)\n", us, allocs, N);
  ::testing::Test::RecordProperty("tick_us", std::to_string(us));
  ::testing::Test::RecordProperty("allocs_per_tick", std::to_string(allocs));
  EXPECT_EQ(f.ctx.behavior_name, "turn_right");
  EXPECT_LT(us, 500.0) << "20 Hz 주기(50 ms)의 1 % 를 넘으면 회귀로 본다";
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
