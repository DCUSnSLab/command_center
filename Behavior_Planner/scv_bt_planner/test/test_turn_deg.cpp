#include <gtest/gtest.h>

#include "scv_bt_planner/bt_nodes.hpp"

using scv_bt_planner::Context;
using scv_bt_planner::PathNode;
using scv_bt_planner::Pose2D;

namespace {
// 원점 datum 이 (1000, 2000) 이라 가정: map = UTM - datum
constexpr double E0 = 1000.0, N0 = 2000.0;

Context makeCtx(std::vector<PathNode> nodes)
{
  Context c;
  c.path.setPath(std::move(nodes), "p");
  return c;
}
}  // namespace

// 10/21 16:24 출발점 실측: 차량 map(-1.08, 4.69) yaw 38.3°, GPS_START = N0426→N0425 링크 82% 투영점(차량과 1.7 m),
// 다음 노드 N0425(-2.04, 0.83). 차량은 실제로 시계방향 약 160° 돌아 남서를 봤다 → 우회전 약 -167°.
// 이전 구현(GPS_START heading 0)은 -129°, 차량→노드 벡터는 측면 오차로 좌회전이 나왔다.
TEST(TurnDeg, GpsStartUsesVehicleYaw1021)
{
  const double sx = 10.296 + 0.82 * (-2.042 - 10.296), sy = 16.011 + 0.82 * (0.826 - 16.011);
  auto c = makeCtx({{"GPS_START", E0 + sx, N0 + sy, 1, 0.0}, {"N0425", E0 - 2.042, N0 + 0.826, 1, 42.5}});
  c.pose = Pose2D{-1.08, 4.69, 38.3 * M_PI / 180.0};
  c.pose_utm = std::make_pair(E0 - 1.08, N0 + 4.69);
  const double out = std::atan2(0.826 - sy, -2.042 - sx) * 180.0 / M_PI;   // -129°
  EXPECT_NEAR(c.turnDegAtTarget(), out - 38.3 + 360.0 * (out - 38.3 < -180.0), 1e-6);
  EXPECT_LT(c.turnDegAtTarget(), -160.0);   // 우회전
}

TEST(TurnDeg, GpsStartAlignedIsStraight)
{
  auto c = makeCtx({{"GPS_START", E0, N0, 1, 0.0}, {"N0425", E0 - 10.0, N0 - 12.0, 1, 0.0}});
  c.pose = Pose2D{0.0, 0.0, std::atan2(-12.0, -10.0)};
  c.pose_utm = std::make_pair(E0, N0);
  EXPECT_NEAR(c.turnDegAtTarget(), 0.0, 1e-6);   // heading 0 이면 -130° 로 회전 오판
}

TEST(TurnDeg, NoPoseNoTurn)
{
  auto c = makeCtx({{"GPS_START", E0, N0, 1, 0.0}, {"B", E0, N0 - 10.0, 1, 0.0}});
  EXPECT_DOUBLE_EQ(c.turnDegAtTarget(), 0.0);
}

TEST(TurnDeg, MidPathUsesPreviousNode)
{
  auto c = makeCtx({{"A", 0, 0, 1, 0}, {"B", 10, 0, 1, 0}, {"C", 10, -10, 1, 0}});
  c.path.advanceToNextNode();
  c.pose = Pose2D{0, 0, 2.0};
  EXPECT_NEAR(c.turnDegAtTarget(), -90.0, 1e-6);   // 우회전
}
