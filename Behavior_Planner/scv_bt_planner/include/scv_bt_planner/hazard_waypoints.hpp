// 위험 지대 경유점 재배치 — 목표/다음 경유점이 코스트맵 치명 영역 안이면 경로 옆으로 비켜 놓거나 건너뛴다.
//
// 왜: 경로 그래프는 구덩이·장애물을 모른다. 경유점이 위험물 위에 있으면 제어기는 그 점을 향해 다가가다가
// 도달 허용거리(경유 ~1.6 m) 안에서 '도달' 처리되고, 그 순간 다음 경유점(위험물 바로 뒤)으로 회피 여유 없이
// 직진해 빠진다(2026-10-02 챔버 scen4: BT·simple 모두 첫 구덩이 추락). 경유점 자체를 빈 곳으로 옮기면
// 제어기는 원래 하던 대로 점을 따라가면서 위험물을 비켜 간다.
//
// 규칙(매 틱, 현재 목표부터 lookahead 개) — 노드마다 경로 법선 방향 '횡위치'를 정하는 프로파일 계획:
//   0. 차량에서 [min_dist, max_dist] 밖의 노드는 판정하지 않는다(기존 오프셋 유지, skip 없음).
//   1. 노드별 필요한 횡위치: 기존 오프셋 점이 비어 있으면 유지(지그재그 방지) → 원위치가 비면 0 →
//      아니면 step 씩 max_shift 까지 직전에 비켜 간 쪽 먼저. '비어 있음' = 점 주변 clear_radius 원(8방위·2단)에 위험 셀 없음.
//   2. 뒤에서 앞으로: 노드 k 에 닿으려면 앞 노드 k-1 의 횡위치가 |Δ| <= y_max(간격) 이어야 한다 — 모자라면
//      앞 노드를 미리 그쪽으로 당긴다(2 m 간격에 0.65 m 씩, 구덩이 위 노드 -1.35 m 를 위해 앞 노드가 -0.7 m).
//   3. 차량에서 앞으로: 차량 위치에서 순서대로 닿을 수 있는지 검사. 빈 곳이 없거나 닿을 수 없으면 skip 표시 —
//      호출자가 현재 목표일 때만, 그리고 건너뛴 다음 목표까지 직선이 통행 가능할 때만 건너뛴다(아니면 정지 유지).
// 코스트맵 밖·미지(-1) 셀은 통행 가능으로 본다(합류 approach_clear 와 같은 fail-open).
#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "scv_bt_planner/path_manager.hpp"

namespace scv_bt_planner {

struct HazardParams {
  double clear_radius = 0.75;   // 차체 반폭 0.37 + 여유 — 점 주변 이 반경 안에 치명 셀이 없어야 한다 [m]
  double max_shift = 2.0;       // 경로 법선 방향 최대 이동 [m]
  double step = 0.25;           // 탐색 간격 [m]
  int lookahead = 3;            // 현재 목표 뒤로 검사할 노드 수
  // 차량에서 이 거리 범위 안의 노드만 판정한다. 가까운 노드는 곧 도달 처리되고(차체 주변 코스트가 섞인다),
  // 먼 노드는 국소 코스트맵 가장자리(감지 범위 경계에 치명 고리가 생긴다, 10/02 실측 ~8 m)라 믿을 수 없다.
  double min_dist = 1.5;
  double max_dist = 6.5;
  // 기구학적 가능성: 직전 점(첫 노드는 차량)에서 진행 거리 d 안에 낼 수 있는 횡이동 = S자(반경 R 두 호)
  //   d < 2R: y_max = 2R(1 - cos(asin(d/2R))),  d >= 2R: 제한 없음.  R=1.7: d 2 m→0.65, 2.6 m→1.21, 3 m→1.80
  // 이보다 큰 재배치는 제어기가 따라갈 수 없다(10/02 scen8: 1 m 앞 노드를 1.25 m 옆으로 → 정지).
  double turn_radius = 1.7;      // Hunter 2.0 최소 회전반경 (smppi min_turning_radius). 0 이면 검사 안 함
  double feasible_factor = 0.9;  // y_max 의 이 비율까지만 쓴다
};

struct HazardPlacement {
  std::string id;
  double dx = 0.0, dy = 0.0;    // 절대 UTM 오프셋
  bool skip = false;            // 빈 곳 없음
  bool changed = false;         // 이번 계산에서 오프셋이 바뀜
  bool out_of_range = false;    // 판정 범위 밖(판정 안 함)
};

class HazardWaypoints {
public:
  using BlockedAt = std::function<bool(double, double)>;                      // 절대 UTM 점이 치명인가
  using SegmentClear = std::function<bool(double, double, double, double)>;   // 절대 UTM 직선이 통행 가능한가

  explicit HazardWaypoints(HazardParams p = {}) : p_(p) {}
  void setParams(const HazardParams& p) { p_ = p; }
  const HazardParams& params() const { return p_; }

  // upcoming: 현재 목표부터의 노드들(앞에서 lookahead+1 개만 본다). prev_node: 현재 목표 직전 노드(방향 계산용, 없으면 nullptr).
  // (sx, sy): 차량 절대 UTM 위치.
  std::vector<HazardPlacement> plan(const std::vector<PathNode>& upcoming, const PathNode* prev_node,
                                    double sx, double sy, const BlockedAt& blocked);

  // 발행용 오프셋(노드 ID → 절대 UTM dx, dy). 없으면 원위치.
  std::pair<double, double> offset(const std::string& id) const;
  // 진행 거리 d 안에서 낼 수 있는 최대 횡이동 [m]
  static double maxLateral(double d, double turn_radius);
  const std::map<std::string, std::pair<double, double>>& offsets() const { return off_; }
  void clear() { off_.clear(); pref_side_ = 0; }

private:
  bool pointFree(double x, double y, const BlockedAt& blocked) const;
  HazardParams p_;
  std::map<std::string, std::pair<double, double>> off_;
  int pref_side_ = 0;   // +1 왼쪽, -1 오른쪽, 0 미정
};

}  // namespace scv_bt_planner
