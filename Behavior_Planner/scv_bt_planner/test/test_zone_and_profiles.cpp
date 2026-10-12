#include <gtest/gtest.h>

#include "scv_bt_planner/profiles.hpp"
#include "scv_bt_planner/zone_table.hpp"

using scv_bt_planner::Profiles;
using scv_bt_planner::ZoneTable;

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

static const std::string DATA = TEST_DATA_DIR;
// 패키지 config 는 test/data 의 한 단계 위 상위에 있다
static const std::string PKG = DATA + "/../..";

TEST(ZoneTable, LoadsAndDefaults)
{
  ZoneTable z;
  std::string err;
  ASSERT_TRUE(z.loadFromFile(DATA + "/graph_zones.json", &err)) << err;
  EXPECT_EQ(z.nodeCount(), 4u);
  EXPECT_EQ(z.size(), 3u);
  EXPECT_EQ(z.zoneOf("N0001"), "sidewalk");    // 미지정 → 기본
  EXPECT_EQ(z.zoneOf("N0002"), "road");
  EXPECT_EQ(z.zoneOf("N0003"), "crosswalk");
  EXPECT_EQ(z.zoneOf("N0004"), "gps_denied");
  EXPECT_EQ(z.zoneOf("NOPE"), "sidewalk");
  EXPECT_TRUE(z.isKnownZone("shared_road"));
  EXPECT_FALSE(z.isKnownZone("moon"));
}

TEST(ZoneTable, BadInput)
{
  ZoneTable z;
  std::string err;
  EXPECT_FALSE(z.loadFromString("not json", &err));
  EXPECT_FALSE(z.loadFromFile(DATA + "/missing.json", &err));
  EXPECT_EQ(z.zoneOf("N0002"), "sidewalk");
}

TEST(Profiles, BaselineFromSmppiYaml)
{
  Profiles p;
  std::string err;
  ASSERT_TRUE(p.loadBaseline(DATA + "/smppi_baseline_test.yaml", &err)) << err;
  EXPECT_DOUBLE_EQ(p.baseline().at("max_linear_velocity"), 1.0);
  EXPECT_DOUBLE_EQ(p.baseline().at("min_linear_velocity"), -0.5);
  EXPECT_DOUBLE_EQ(p.baseline().at("lookahead_base_distance"), 10.0);
  EXPECT_DOUBLE_EQ(p.baseline().at("goal_reached_threshold"), 3.0);
  EXPECT_DOUBLE_EQ(p.baseline().at("control_frequency"), 10.0);
  EXPECT_EQ(p.baseline().count("xy_goal_tolerance"), 0u) << "원본과 같이 파일 경로에서는 뽑지 않는다";
}

TEST(Profiles, NodeProfileParityWithSimpleBp)
{
  Profiles p;
  std::string err;
  ASSERT_TRUE(p.loadProfiles(PKG + "/config/behavior_profiles.yaml", &err)) << err;
  ASSERT_TRUE(p.loadBaseline(DATA + "/smppi_baseline_test.yaml", &err)) << err;
  EXPECT_EQ(p.nodeProfileCount(), 11u);

  std::string desc; int code = 0;
  // type 1: baseline 그대로
  auto t1 = p.compute(1, "", &desc, &code);
  EXPECT_DOUBLE_EQ(t1.at("max_linear_velocity"), 1.0);
  EXPECT_EQ(code, 1);
  EXPECT_EQ(desc, "Normal forward movement");
  // type 2 후진: goal_weight ×1.7, lookahead ×0.6, override max 0 / min -1 / reverse true
  auto t2 = p.compute(2, "", &desc, &code);
  EXPECT_DOUBLE_EQ(t2.at("goal_weight"), 51.0);
  EXPECT_DOUBLE_EQ(t2.at("lookahead_base_distance"), 6.0);
  EXPECT_DOUBLE_EQ(t2.at("max_linear_velocity"), 0.0);
  EXPECT_DOUBLE_EQ(t2.at("min_linear_velocity"), -1.0);
  EXPECT_DOUBLE_EQ(t2.at("respect_reverse_heading"), 1.0);
  EXPECT_EQ(t2.count("xy_goal_tolerance"), 0u) << "baseline 에 없는 키의 multiplier 는 무시";
  // type 3 정밀 전진: max_v ×0.66, goal_reached_threshold ×0.5
  auto t3 = p.compute(3, "");
  EXPECT_NEAR(t3.at("max_linear_velocity"), 0.66, 1e-9);
  EXPECT_DOUBLE_EQ(t3.at("goal_reached_threshold"), 1.5);
  // 미지 유형 → 1
  auto t99 = p.compute(99, "", &desc, &code);
  EXPECT_DOUBLE_EQ(t99.at("max_linear_velocity"), 1.0);
  EXPECT_EQ(code, 1);
  EXPECT_TRUE(Profiles::validate(t1));
  EXPECT_TRUE(Profiles::validate(t2));
}

TEST(Profiles, OverlayLayersOnTop)
{
  Profiles p;
  ASSERT_TRUE(p.loadProfiles(PKG + "/config/behavior_profiles.yaml"));
  ASSERT_TRUE(p.loadBaseline(DATA + "/smppi_baseline_test.yaml"));
  std::string desc; int code = 0;
  // sidewalk 오버레이 = 무변경 (동등성 기준)
  auto s = p.compute(1, "sidewalk", &desc, &code);
  EXPECT_DOUBLE_EQ(s.at("max_linear_velocity"), 1.0);
  EXPECT_EQ(code, 20);
  // 정밀 전진 + 좌회전: 0.66 × 0.6
  auto tl = p.compute(3, "turn_left", &desc, &code);
  EXPECT_NEAR(tl.at("max_linear_velocity"), 0.66 * 0.6, 1e-9);
  EXPECT_EQ(code, 25);
  EXPECT_NE(desc.find("Turn left"), std::string::npos);
  // 보행자 혼재: obstacle_weight ×1.5
  auto sh = p.compute(1, "shared_road");
  EXPECT_DOUBLE_EQ(sh.at("obstacle_weight"), 150.0);
  EXPECT_DOUBLE_EQ(sh.at("max_linear_velocity"), 0.5);
  // 미지 오버레이 → 무시
  auto un = p.compute(1, "moon", &desc, &code);
  EXPECT_DOUBLE_EQ(un.at("max_linear_velocity"), 1.0);
  EXPECT_EQ(code, 1);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
