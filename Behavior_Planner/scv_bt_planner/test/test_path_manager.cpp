// simple_behavior_planner/test/test_join_rule.py + test_join_guards.py 의 gtest 이식.
// 케이스·기하·기대값을 원본과 같게 유지한다 (동작 동등성 회귀).
#include <cmath>
#include <gtest/gtest.h>

#include "scv_bt_planner/path_manager.hpp"

using scv_bt_planner::PathManager;
using scv_bt_planner::PathNode;

namespace {

PathManager mk(const std::vector<std::pair<double, double>>& coords)
{
  std::vector<PathNode> nodes;
  int i = 0;
  for (auto [x, y] : coords) {
    PathNode n; n.id = "N" + std::to_string(i++); n.x = x; n.y = y;
    nodes.push_back(n);
  }
  PathManager pm;
  pm.setPath(nodes, "test");
  return pm;
}

// 2026-08-05 실측 기하
const std::vector<std::pair<double, double>> FIELD = {
  {36.9, 33.2}, {33.0, 36.5}, {29.1, 39.7}, {25.6, 43.3}, {21.8, 46.6},
  {17.4, 49.0}, {13.4, 52.2}, {10.0, 55.9}, {6.7, 59.7}, {3.1, 63.3},
  {0.2, 67.5}, {-2.3, 71.9}};
constexpr double VX = 55.2, VY = 40.8;

double total(const PathManager& pm, int i)
{
  const auto& n = pm.nodes();
  double d = std::hypot(n[i].x - VX, n[i].y - VY);
  for (size_t k = i; k + 1 < n.size(); ++k) d += std::hypot(n[k + 1].x - n[k].x, n[k + 1].y - n[k].y);
  return d;
}

}  // namespace

TEST(JoinRule, CostRuleBeatsNearest)
{
  auto pm = mk(FIELD);
  const int idx_cost = pm.alignToPosition(VX, VY);
  auto pm2 = mk(FIELD);
  const int idx_near = pm2.alignToPosition(VX, VY, 0.0);
  EXPECT_EQ(idx_near, 0) << "최근접 규칙은 경로 꼬리를 고른다";
  EXPECT_GT(idx_cost, idx_near) << "비용 규칙은 더 앞쪽 노드를 고른다";
  EXPECT_LT(total(pm, idx_cost), total(pm2, idx_near) - 1.0);

  const auto& n = pm.nodes()[idx_cost];
  const double br = std::atan2(n.y - VY, n.x - VX) * 180.0 / M_PI;
  const auto& goal = pm.nodes().back();
  const double gbr = std::atan2(goal.y - VY, goal.x - VX) * 180.0 / M_PI;
  const double diff = std::fabs(std::fmod(br - gbr + 540.0, 360.0) - 180.0);
  EXPECT_LT(diff, 90.0) << "합류 방향이 목표 방향과 90도 이내";
}

TEST(JoinRule, ApproachCap)
{
  auto pm = mk(FIELD);
  const int idx_cost = pm.alignToPosition(VX, VY);
  auto pm3 = mk(FIELD);
  const int i3 = pm3.alignToPosition(VX, VY, 30.0);
  const double d3 = std::hypot(pm3.nodes()[i3].x - VX, pm3.nodes()[i3].y - VY);
  EXPECT_LE(d3, 30.0 + 1e-6);
  EXPECT_LE(i3, idx_cost);
}

TEST(JoinRule, UnweightedCrossesPath)
{
  auto pm = mk(FIELD);
  const int i_w1 = pm.alignToPosition(35.0, 34.0, 40.0, 1.0, 99);
  EXPECT_GT(i_w1, 2) << "무가중(1.0)은 경로 머리 옆에서도 앞쪽으로 건너뜀";
}

TEST(JoinRule, NearHeadJoinsWithinOneNode)
{
  auto pm4 = mk(FIELD);
  const int i4 = pm4.alignToPosition(35.0, 34.0);
  auto pm5 = mk(FIELD);
  const int i5 = pm5.alignToPosition(35.0, 34.0, 0.0);
  EXPECT_LE(i4, i5 + 1);
}

TEST(JoinRule, MidPathDoesNotGoBack)
{
  auto pm6 = mk(FIELD);
  EXPECT_GE(pm6.alignToPosition(13.0, 53.0), 5);
}

TEST(JoinRule, AllOverCapFallsBackToNearest)
{
  auto pm7 = mk(FIELD);
  const int i7 = pm7.alignToPosition(500.0, 500.0, 10.0);
  int near = 0; double best = 1e300;
  for (size_t k = 0; k < FIELD.size(); ++k) {
    const double d2 = std::pow(FIELD[k].first - 500.0, 2) + std::pow(FIELD[k].second - 500.0, 2);
    if (d2 < best) { best = d2; near = static_cast<int>(k); }
  }
  EXPECT_EQ(i7, near);
}

TEST(JoinRule, StraightPathSkipLimited)
{
  std::vector<std::pair<double, double>> straight;
  for (int i = 0; i < 8; ++i) straight.push_back({3.0 * i, 0.0});
  auto pm8 = mk(straight);
  const int i8 = pm8.alignToPosition(10.0, 8.0);
  auto pm9 = mk(straight);
  const int i9 = pm9.alignToPosition(10.0, 8.0, 25.0, 1.5, 99);
  EXPECT_LE(i8, 5);
  EXPECT_LT(i8, i9);
}

// ---- test_join_guards.py --------------------------------------------------

namespace {
std::vector<std::pair<double, double>> fieldGuard()
{
  std::vector<std::pair<double, double>> v;
  for (int i = 0; i < 20; ++i) v.push_back({36.1 + 1.3 * i, 35.5 + 1.3 * i});
  return v;
}
const std::vector<std::pair<double, double>> CHAMBER = {{0, 1}, {0, 3}, {0, 5}, {3, 5}};
constexpr double V_START_X = 35.0, V_START_Y = 34.5, V_ENGAGE_X = 48.0, V_ENGAGE_Y = 47.5;
constexpr double V_CH_X = 6.0, V_CH_Y = 1.0;

PathManager chamberWithHistory()
{
  auto pm = mk(CHAMBER);
  for (int k = 0; k < 13; ++k) pm.notePosition(0.5 * k, 1.0);
  return pm;
}

bool chamberClear(double x0, double y0, double x1, double y1)
{
  const double OBS[3][2] = {{4.75, 2.25}, {4.75, 2.75}, {5.25, 2.25}};
  const double margin = 0.45, step = 0.25;
  const double d = std::hypot(x1 - x0, y1 - y0);
  for (int k = 0; k <= static_cast<int>(d / step); ++k) {
    const double t = d > 0 ? (k * step) / d : 0.0;
    const double px = x0 + (x1 - x0) * t, py = y0 + (y1 - y0) * t;
    for (const auto& o : OBS) {
      if (std::fabs(px - o[0]) < 0.25 + margin && std::fabs(py - o[1]) < 0.25 + margin) return false;
    }
  }
  return true;
}
}  // namespace

TEST(JoinGuards, FieldCatchUpAfterManualDrive)
{
  auto pm = mk(fieldGuard());
  const int i_start = pm.alignToPosition(V_START_X, V_START_Y);
  const int i_engage = pm.alignToPosition(V_ENGAGE_X, V_ENGAGE_Y);
  EXPECT_GT(i_engage, i_start + 3);

  auto pm2 = mk(fieldGuard());
  pm2.alignToPosition(V_START_X, V_START_Y);
  const int i_bound = pm2.alignToPosition(V_ENGAGE_X, V_ENGAGE_Y, 25.0, 1.5, 2, 2);
  EXPECT_LE(i_bound, i_start + 2) << "현재기준 상한 2 를 켜면 따라잡기가 막힌다";
}

TEST(JoinGuards, ChamberShortcut)
{
  auto pm3 = mk(CHAMBER);
  EXPECT_EQ(pm3.alignToPosition(V_CH_X, V_CH_Y, 25.0, 1.5, 2, 0, nullptr, false), 3)
    << "상한 없으면 경로 끝(P3)으로 건너뛴다";
  EXPECT_LT(chamberWithHistory().alignToPosition(V_CH_X, V_CH_Y), 3)
    << "기본값에서는 건너뛰지 않는다";
  auto pm4 = mk(CHAMBER);
  EXPECT_LE(pm4.alignToPosition(V_CH_X, V_CH_Y, 25.0, 1.5, 2, 2, nullptr, false), 2);
}

TEST(JoinGuards, ApproachClear)
{
  EXPECT_FALSE(chamberClear(6, 1, 3, 5));
  EXPECT_TRUE(chamberClear(6, 1, 0, 1));
  auto pm5 = mk(CHAMBER);
  EXPECT_LT(pm5.alignToPosition(V_CH_X, V_CH_Y, 25.0, 1.5, 2, 0, chamberClear), 3);

  auto pm = mk(fieldGuard());
  const int i_engage = [&] { auto t = mk(fieldGuard()); t.alignToPosition(V_START_X, V_START_Y);
                             return t.alignToPosition(V_ENGAGE_X, V_ENGAGE_Y); }();
  pm.alignToPosition(V_START_X, V_START_Y);
  const int i_f3 = pm.alignToPosition(V_ENGAGE_X, V_ENGAGE_Y, 25.0, 1.5, 2, 0,
                                      [](double, double, double, double) { return true; });
  EXPECT_EQ(i_f3, i_engage);

  const int i_blk = chamberWithHistory().alignToPosition(
    V_CH_X, V_CH_Y, 25.0, 1.5, 2, 0, [](double, double, double, double) { return false; });
  EXPECT_LE(i_blk, 2) << "전 후보 차단 시에도 합류점은 게이트 안";
  auto pm7b = mk(CHAMBER);
  EXPECT_EQ(pm7b.alignToPosition(V_CH_X, V_CH_Y, 25.0, 1.5, 2, 0,
                                 [](double, double, double, double) { return false; }, false), 3);
}

TEST(JoinGuards, PassedHistoryGate)
{
  auto pm8 = mk(fieldGuard());
  const int i_start = pm8.alignToPosition(V_START_X, V_START_Y);
  for (int k = 0; k <= 20; ++k) {
    const double t = k / 20.0;
    pm8.notePosition(V_START_X + (V_ENGAGE_X - V_START_X) * t, V_START_Y + (V_ENGAGE_Y - V_START_Y) * t);
  }
  EXPECT_GE(pm8.alignToPosition(V_ENGAGE_X, V_ENGAGE_Y), i_start + 3);

  auto pm9 = chamberWithHistory();
  const int i_c9 = pm9.alignToPosition(V_CH_X, V_CH_Y);
  EXPECT_LE(i_c9, 2) << "지나온 다음 칸까지만 허용 (P2 상한)";
}

TEST(PathManager, AdvanceAndNext)
{
  auto pm = mk(FIELD);
  EXPECT_EQ(pm.nextNodes(3).size(), 3u);
  for (int i = 0; i < 11; ++i) EXPECT_TRUE(pm.advanceToNextNode());
  EXPECT_TRUE(pm.isFinalNode());
  EXPECT_TRUE(pm.nextNodes(3).empty());
  EXPECT_FALSE(pm.advanceToNextNode());
  EXPECT_FALSE(pm.isFollowing());
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
