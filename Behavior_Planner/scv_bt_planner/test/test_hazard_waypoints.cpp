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
  double y_left = 2.0, y_right = -3.0;   // 보도 밖(연석 너머)도 치명
  bool blocked(double x, double y) const
  {
    if (y > y_left || y < y_right) return true;
    for (const auto& r : lethal) {
      if (x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) return true;
    }
    return false;
  }
  bool seg(double x0, double y0, double x1, double y1) const
  {
    const int n = 50;
    for (int k = 0; k <= n; ++k) {
      const double t = static_cast<double>(k) / n;
      if (blocked(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)) return false;
    }
    return true;
  }
};

std::vector<HazardPlacement> run(HazardWaypoints& h, const World& w, const std::vector<PathNode>& up, double sx,
                                 double sy)
{
  // 기존 시험은 판정 범위 제한 없이 본다(범위 시험은 따로)
  HazardParams p = h.params();
  if (p.max_dist == HazardParams{}.max_dist && p.min_dist == HazardParams{}.min_dist) {
    p.min_dist = 0.0; p.max_dist = 1e9; h.setParams(p);
  }
  return h.plan(up, nullptr, sx, sy, [&](double x, double y) { return w.blocked(x, y); },
                [&](double a, double b, double c, double d) { return w.seg(a, b, c, d); });
}

}  // namespace

TEST(HazardWaypoints, ClearPathKeepsNodes)
{
  World w;
  HazardWaypoints h;
  auto pl = run(h, w, line(5), -2.0, 0.0);
  ASSERT_EQ(pl.size(), 4u);   // 현재 + lookahead 3
  for (const auto& p : pl) {
    EXPECT_FALSE(p.skip);
    EXPECT_DOUBLE_EQ(p.dx, 0.0);
    EXPECT_DOUBLE_EQ(p.dy, 0.0);
  }
}

TEST(HazardWaypoints, NodeInPitIsShiftedRightBeyondClearance)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});   // 구덩이, 노드 H6(x=12) 이 안에 있다
  HazardWaypoints h;
  std::vector<PathNode> all = line(10);
  std::vector<PathNode> up(all.begin() + 5, all.end());   // 목표 H5(x=10)
  auto pl = run(h, w, up, 8.0, 0.0);
  ASSERT_GE(pl.size(), 2u);
  EXPECT_DOUBLE_EQ(pl[0].dy, 0.0);            // H5 는 그대로
  EXPECT_FALSE(pl[1].skip);
  EXPECT_LT(pl[1].dy, -0.6 - 0.75 + 1e-9);    // 오른쪽(우측통행 기본)으로 구덩이 가장자리+반경 밖
  EXPECT_GE(pl[1].dy, -2.0);
  EXPECT_FALSE(w.blocked(12.0 + pl[1].dx, pl[1].dy));
}

TEST(HazardWaypoints, ConsecutivePitsKeepSameSide)
{
  World w;
  for (double c : {12.0, 16.0, 20.0}) w.lethal.push_back({c - 0.5, c + 0.5, -0.6, 0.6});
  HazardWaypoints h;
  std::vector<PathNode> all = line(14);
  HazardParams p; p.lookahead = 8; p.min_dist = 0.0; p.max_dist = 1e9;
  h.setParams(p);
  std::vector<PathNode> up(all.begin() + 5, all.end());
  auto pl = run(h, w, up, 9.0, 0.0);
  // H6, H8, H10 (구덩이 위) 모두 같은 쪽(오른쪽)으로
  for (int idx : {1, 3, 5}) {
    EXPECT_FALSE(pl[idx].skip) << idx;
    EXPECT_LT(pl[idx].dy, -1.3) << idx;
  }
}

TEST(HazardWaypoints, HysteresisKeepsExistingOffset)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});
  HazardWaypoints h;
  std::vector<PathNode> all = line(10);
  std::vector<PathNode> up(all.begin() + 6, all.end());   // 목표 H6
  auto a = run(h, w, up, 9.0, 0.0);
  ASSERT_FALSE(a[0].skip);
  const double dy0 = a[0].dy;
  w.lethal.clear();                                          // 코스트맵에서 구덩이가 잠깐 사라져도
  auto b = run(h, w, up, 9.2, 0.0);
  EXPECT_DOUBLE_EQ(b[0].dy, dy0);                            // 기존 오프셋 유지(유효한 동안)
  EXPECT_FALSE(b[0].changed);
}

TEST(HazardWaypoints, NoFreeSpotMarksSkip)
{
  World w;
  w.lethal.push_back({11.0, 13.0, -3.5, 2.5});   // 보도 전폭을 막는 위험물
  HazardWaypoints h;
  std::vector<PathNode> all = line(10);
  std::vector<PathNode> up(all.begin() + 6, all.end());
  auto pl = run(h, w, up, 9.0, 0.0);
  EXPECT_TRUE(pl[0].skip);
  EXPECT_DOUBLE_EQ(h.offset("H6").second, 0.0);
}

TEST(HazardWaypoints, RestoresWhenOffsetBecomesInvalid)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});
  HazardWaypoints h;
  std::vector<PathNode> all = line(10);
  std::vector<PathNode> up(all.begin() + 6, all.end());
  auto a = run(h, w, up, 9.0, 0.0);
  ASSERT_LT(a[0].dy, 0.0);
  // 오른쪽에 새 장애물이 생겨 기존 오프셋 점이 막히면 다시 찾는다(왼쪽 또는 원위치)
  w.lethal.clear();
  w.lethal.push_back({11.0, 13.0, -3.0, -0.2});
  auto b = run(h, w, up, 9.0, 0.0);
  EXPECT_FALSE(b[0].skip);
  EXPECT_TRUE(b[0].changed);
  EXPECT_FALSE(w.blocked(12.0 + b[0].dx, b[0].dy));
}

TEST(HazardWaypoints, OnlyNodesInDistanceWindowAreJudged)
{
  World w;
  w.lethal.push_back({-0.5, 0.5, -0.6, 0.6});    // 차량 바로 앞(가까운 노드 H0)
  w.lethal.push_back({13.5, 14.5, -0.6, 0.6});   // 먼 노드 H7
  HazardWaypoints h;   // 기본 범위 1.5~6.5 m
  HazardParams p; p.lookahead = 8; h.setParams(p);
  auto pl = h.plan(line(9), nullptr, -1.0, 0.0, [&](double x, double y) { return w.blocked(x, y); },
                   [&](double a, double b, double c, double d) { return w.seg(a, b, c, d); });
  EXPECT_TRUE(pl[0].out_of_range);   // 1.0 m — 곧 도달
  EXPECT_FALSE(pl[0].skip);
  EXPECT_TRUE(pl[7].out_of_range);   // 15 m — 국소 코스트맵 가장자리
  EXPECT_DOUBLE_EQ(pl[7].dy, 0.0);
  EXPECT_FALSE(pl[2].out_of_range);  // 5 m
}

TEST(HazardWaypoints, KeepsOffsetWhilePointStillFreeEvenIfSegmentBlocked)
{
  World w;
  w.lethal.push_back({11.5, 12.5, -0.6, 0.6});
  HazardWaypoints h;
  std::vector<PathNode> all = line(10);
  std::vector<PathNode> up(all.begin() + 6, all.end());
  auto a = run(h, w, up, 8.0, 0.0);
  ASSERT_LT(a[0].dy, 0.0);
  w.lethal.push_back({9.5, 10.5, -2.5, -0.9});   // 차량→재배치 점 직선을 가리는 것이 생겨도 점이 비어 있으면 유지
  auto b = run(h, w, up, 8.0, 0.0);
  EXPECT_DOUBLE_EQ(b[0].dy, a[0].dy);
}
