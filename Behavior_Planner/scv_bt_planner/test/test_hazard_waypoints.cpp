// 위험 지대 경유점 재배치 단위시험 — 챔버 scen_hazard 와 같은 기하(직선 경로, 노드 2 m, 구덩이 1.0x1.2 m).
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "scv_bt_planner/hazard_waypoints.hpp"

using namespace scv_bt_planner;

namespace {

struct Rect { double x0, x1, y0, y1; };

std::vector<PathNode> line(int n, double spacing = 2.0)
{
  std::vector<PathNode> v;
  for (int i = 0; i < n; ++i) v.push_back({"H" + std::to_string(i), i * spacing, 0.0, 1, 0.0});
  return v;
}

struct World {
  std::vector<Rect> lethal;
  double y_left = 2.0, y_right = -3.0;   // 보도 밖(연석 너머)도 위험
  bool blocked(double x, double y) const
  {
    if (y > y_left || y < y_right) return true;
    for (const auto& r : lethal) {
      if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) return true;
    }
    return false;
  }
};

HazardParams wide(int lookahead = 3)
{
  HazardParams p; p.lookahead = lookahead; p.min_dist = 0.0; p.max_dist = 1e9;
  return p;
}

std::vector<HazardPlacement> run(HazardWaypoints& h, const World& w, const std::vector<PathNode>& up, double sx,
                                 double sy)
{
  return h.plan(up, nullptr, sx, sy, [&](double x, double y) { return w.blocked(x, y); });
}

std::vector<PathNode> from(const std::vector<PathNode>& all, int i) { return {all.begin() + i, all.end()}; }

}  // namespace

TEST(HazardWaypoints, MaxLateralMatchesSCurve)
{
  EXPECT_NEAR(HazardWaypoints::maxLateral(2.0, 1.7), 0.65, 0.01);
  EXPECT_NEAR(HazardWaypoints::maxLateral(2.6, 1.7), 1.21, 0.01);
  EXPECT_DOUBLE_EQ(HazardWaypoints::maxLateral(0.0, 1.7), 0.0);
  EXPECT_GT(HazardWaypoints::maxLateral(3.5, 1.7), 100.0);   // 2R 이상이면 제한 없음
  EXPECT_GT(HazardWaypoints::maxLateral(1.0, 0.0), 100.0);   // R=0 이면 검사 안 함
}

TEST(HazardWaypoints, ClearPathKeepsNodes)
{
  World w;
  HazardWaypoints h(wide());
  auto pl = run(h, w, line(5), -2.0, 0.0);
  ASSERT_EQ(pl.size(), 4u);
  for (const auto& p : pl) {
    EXPECT_FALSE(p.skip);
    EXPECT_DOUBLE_EQ(p.dy, 0.0);
  }
}

TEST(HazardWaypoints, PitNodeShiftedAndPreviousNodePulledForReachability)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});   // 구덩이, H6(x=12) 이 안에 있다
  HazardWaypoints h(wide());
  auto pl = run(h, w, from(line(10), 5), 6.0, 0.0);   // 목표 H5(x=10), 차량 4 m 뒤
  ASSERT_GE(pl.size(), 2u);
  EXPECT_FALSE(pl[1].skip);
  EXPECT_LT(pl[1].dy, -0.6 - 0.75 + 1e-9);            // H6: 구덩이 가장자리+반경 밖(오른쪽 기본)
  EXPECT_FALSE(w.blocked(12.0 + pl[1].dx, pl[1].dy));
  EXPECT_FALSE(pl[0].skip);
  EXPECT_LT(pl[0].dy, -0.5);                          // H5: H6 에 닿도록 미리 당겨짐(2 m 에 ≤0.59 m 차이)
  EXPECT_LE(std::fabs(pl[1].dy - pl[0].dy), 0.9 * HazardWaypoints::maxLateral(2.0, 1.7) + 1e-9);
}

TEST(HazardWaypoints, TooLateIsUnreachable)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});
  HazardWaypoints h(wide());
  auto pl = run(h, w, from(line(10), 6), 11.0, 0.0);   // 목표 H6, 차량 1 m 앞 — 1.35 m 비킬 수 없다
  EXPECT_TRUE(pl[0].skip);
  EXPECT_DOUBLE_EQ(h.offset("H6").second, 0.0);
}

TEST(HazardWaypoints, ConsecutivePitsStayOnOneSideWithFeasibleTransitions)
{
  World w;
  for (double c : {12.0, 16.0, 20.0}) w.lethal.push_back({c - 0.5, c + 0.5, -0.6, 0.6});
  HazardWaypoints h(wide(8));
  auto pl = run(h, w, from(line(14), 5), 6.0, 0.0);   // H5..H13
  for (int idx : {1, 3, 5}) {                          // H6, H8, H10
    EXPECT_FALSE(pl[idx].skip) << idx;
    EXPECT_LT(pl[idx].dy, -1.3) << idx;
  }
  const double b = 0.9 * HazardWaypoints::maxLateral(2.0, 1.7);
  for (size_t i = 1; i < pl.size(); ++i) {
    if (pl[i].skip || pl[i - 1].skip) continue;
    EXPECT_LE(std::fabs(pl[i].dy - pl[i - 1].dy), b + 1e-9) << i;   // 이웃 노드 횡변화가 기구학 한도 안
  }
}

TEST(HazardWaypoints, HysteresisKeepsExistingOffset)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});
  HazardWaypoints h(wide());
  auto up = from(line(10), 6);
  auto a = run(h, w, up, 8.0, 0.0);
  ASSERT_FALSE(a[0].skip);
  const double dy0 = a[0].dy;
  w.lethal.clear();                     // 코스트맵에서 구덩이가 잠깐 사라져도
  auto b = run(h, w, up, 8.3, a[0].dy * 0.2);
  EXPECT_DOUBLE_EQ(b[0].dy, dy0);       // 기존 오프셋 점이 비어 있으면 유지
  EXPECT_FALSE(b[0].changed);
}

TEST(HazardWaypoints, NoFreeSpotMarksSkip)
{
  World w;
  w.lethal.push_back({11.0, 13.0, -3.5, 2.5});   // 보도 전폭을 막는 위험물
  HazardWaypoints h(wide());
  auto pl = run(h, w, from(line(10), 6), 8.0, 0.0);
  EXPECT_TRUE(pl[0].skip);
}

TEST(HazardWaypoints, OnlyNodesInDistanceWindowAreJudged)
{
  World w;
  w.lethal.push_back({-0.5, 0.5, -0.6, 0.6});    // 차량 바로 앞(가까운 노드 H0)
  w.lethal.push_back({13.5, 14.5, -0.6, 0.6});   // 먼 노드 H7
  HazardParams p; p.lookahead = 8;               // 기본 범위 1.5~6.5 m
  HazardWaypoints h(p);
  auto pl = run(h, w, line(9), -1.0, 0.0);
  EXPECT_TRUE(pl[0].out_of_range);
  EXPECT_FALSE(pl[0].skip);
  EXPECT_TRUE(pl[7].out_of_range);
  EXPECT_DOUBLE_EQ(pl[7].dy, 0.0);
  EXPECT_FALSE(pl[2].out_of_range);
}
