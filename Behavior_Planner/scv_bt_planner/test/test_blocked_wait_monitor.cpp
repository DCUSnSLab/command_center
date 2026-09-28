#include <gtest/gtest.h>

#include "scv_bt_planner/blocked_wait_monitor.hpp"

using scv_bt_planner::BlockedWaitMonitor;

namespace {
bool has(const std::vector<std::string>& v, const std::string& s)
{
  for (const auto& x : v) if (x == s) return true;
  return false;
}
}

TEST(BlockedWait, EscalatesAndRecovers)
{
  BlockedWaitMonitor m;  // 4/12/10 s 기본
  BlockedWaitMonitor::XY p{0.0, 0.0};
  double t = 0.0;
  auto a = m.update(t, p, true);
  EXPECT_TRUE(a.empty());
  EXPECT_EQ(m.state(), "NORMAL");
  t = 3.9; m.update(t, p, true); EXPECT_EQ(m.state(), "NORMAL");
  t = 4.0; a = m.update(t, p, true);
  EXPECT_EQ(m.state(), "BLOCKED_WAIT");
  EXPECT_TRUE(has(a, "announce:BLOCKED_WAIT"));
  t = 15.9; m.update(t, p, true); EXPECT_EQ(m.state(), "BLOCKED_WAIT");
  t = 16.0; a = m.update(t, p, true);
  EXPECT_EQ(m.state(), "CREEP");
  EXPECT_TRUE(has(a, "creep_on"));
  t = 26.0; a = m.update(t, p, true);
  EXPECT_EQ(m.state(), "ASSIST");
  EXPECT_TRUE(has(a, "creep_off"));
  EXPECT_TRUE(has(a, "assist"));
  t = 40.9; a = m.update(t, p, true); EXPECT_FALSE(has(a, "assist"));
  t = 41.0; a = m.update(t, p, true); EXPECT_TRUE(has(a, "assist")) << "15 s 반복";
  // 진행 재개 → NORMAL
  t = 42.0; a = m.update(t, BlockedWaitMonitor::XY{1.0, 0.0}, true);
  EXPECT_EQ(m.state(), "NORMAL");
  EXPECT_TRUE(has(a, "announce:NORMAL"));
}

TEST(BlockedWait, PlannedStopIsNotBlocked)
{
  BlockedWaitMonitor m;
  BlockedWaitMonitor::XY p{0.0, 0.0};
  for (double t = 0; t < 60; t += 0.5) {
    m.update(t, p, false);
    EXPECT_EQ(m.state(), "NORMAL");
  }
}

TEST(BlockedWait, CreepClearsOnProgress)
{
  BlockedWaitMonitor m;
  BlockedWaitMonitor::XY p{0.0, 0.0};
  m.update(0.0, p, true);
  m.update(4.0, p, true);
  auto a = m.update(16.0, p, true);
  ASSERT_EQ(m.state(), "CREEP");
  a = m.update(17.0, BlockedWaitMonitor::XY{0.2, 0.0}, true);
  EXPECT_EQ(m.state(), "NORMAL");
  EXPECT_TRUE(has(a, "creep_off"));
}

TEST(BlockedWait, UnknownPositionNoop)
{
  BlockedWaitMonitor m;
  auto a = m.update(0.0, std::nullopt, true);
  EXPECT_TRUE(a.empty());
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
