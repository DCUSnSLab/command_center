#include "scv_bt_planner/hazard_waypoints.hpp"

#include <cmath>

namespace scv_bt_planner {

bool HazardWaypoints::pointFree(double x, double y, const BlockedAt& blocked) const
{
  if (blocked(x, y)) return false;
  for (double r : {p_.clear_radius * 0.5, p_.clear_radius}) {
    for (int k = 0; k < 8; ++k) {
      const double a = k * M_PI / 4.0;
      if (blocked(x + r * std::cos(a), y + r * std::sin(a))) return false;
    }
  }
  return true;
}

std::pair<double, double> HazardWaypoints::offset(const std::string& id) const
{
  auto it = off_.find(id);
  return it == off_.end() ? std::make_pair(0.0, 0.0) : it->second;
}

std::vector<HazardPlacement> HazardWaypoints::plan(const std::vector<PathNode>& upcoming, const PathNode* prev_node,
                                                   double sx, double sy, const BlockedAt& blocked,
                                                   const SegmentClear& seg)
{
  std::vector<HazardPlacement> out;
  const size_t n = std::min(upcoming.size(), static_cast<size_t>(std::max(0, p_.lookahead) + 1));
  double px = sx, py = sy;   // 직전 점(차량 → 재배치된 노드 순)
  for (size_t i = 0; i < n; ++i) {
    const PathNode& nd = upcoming[i];
    // 경로 진행 방향: 다음 노드가 있으면 (이 노드→다음), 없으면 (이전→이 노드)
    double tx = 0.0, ty = 0.0;
    if (i + 1 < upcoming.size()) {
      tx = upcoming[i + 1].x - nd.x; ty = upcoming[i + 1].y - nd.y;
    } else if (i > 0) {
      tx = nd.x - upcoming[i - 1].x; ty = nd.y - upcoming[i - 1].y;
    } else if (prev_node) {
      tx = nd.x - prev_node->x; ty = nd.y - prev_node->y;
    } else {
      tx = nd.x - sx; ty = nd.y - sy;
    }
    const double tl = std::hypot(tx, ty);
    const double nx = tl > 1e-6 ? -ty / tl : 0.0, ny = tl > 1e-6 ? tx / tl : 1.0;   // 왼쪽 법선

    HazardPlacement pl;
    pl.id = nd.id;
    const auto prev_off = offset(nd.id);
    const double dveh = std::hypot(nd.x - sx, nd.y - sy);
    if (dveh < p_.min_dist || dveh > p_.max_dist) {
      // 판정 범위 밖: 기존 오프셋 그대로(변경 없음). 직전 점은 이 노드(재배치 반영)로 이어 간다.
      pl.dx = prev_off.first; pl.dy = prev_off.second; pl.out_of_range = true;
      px = nd.x + prev_off.first; py = nd.y + prev_off.second;
      out.push_back(pl);
      continue;
    }
    const bool from_vehicle = (px == sx && py == sy);
    auto valid = [&](double qx, double qy, bool need_seg) {
      if (!pointFree(qx, qy, blocked)) return false;
      if (!need_seg) return true;
      double ax = px, ay = py;
      if (from_vehicle) {   // 차량 주변은 차체·팽창 코스트가 섞인다 — 앞쪽부터 본다
        const double d = std::hypot(qx - px, qy - py);
        if (d > p_.seg_skip_from_vehicle) {
          ax = px + (qx - px) * p_.seg_skip_from_vehicle / d;
          ay = py + (qy - py) * p_.seg_skip_from_vehicle / d;
        } else {
          return true;
        }
      }
      return seg(ax, ay, qx, qy);
    };
    bool found = false;
    double ox = 0.0, oy = 0.0;
    // 1) 기존 오프셋 점이 아직 비어 있으면 유지
    if (prev_off.first != 0.0 || prev_off.second != 0.0) {
      if (valid(nd.x + prev_off.first, nd.y + prev_off.second, false)) {
        ox = prev_off.first; oy = prev_off.second; found = true;
      }
    }
    // 2) 원위치, 3) 법선 탐색 — 직선 조건 포함 → 점 조건만
    for (int pass = 0; pass < 2 && !found; ++pass) {
      const bool need_seg = (pass == 0);
      if (valid(nd.x, nd.y, need_seg)) { ox = oy = 0.0; found = true; break; }
      const int first = pref_side_ != 0 ? pref_side_ : -1;   // 미정이면 오른쪽 먼저(우측통행)
      for (double s = p_.step; s <= p_.max_shift + 1e-9 && !found; s += p_.step) {
        for (int side : {first, -first}) {
          const double qx = nd.x + side * s * nx, qy = nd.y + side * s * ny;
          if (valid(qx, qy, need_seg)) { ox = qx - nd.x; oy = qy - nd.y; found = true; break; }
        }
      }
    }
    if (!found) {
      pl.skip = true;
      pl.changed = off_.erase(nd.id) > 0;
      out.push_back(pl);
      continue;   // 직전 점은 그대로 — 다음 노드는 마지막으로 놓인 점에서 이어 본다
    }
    pl.dx = ox; pl.dy = oy;
    pl.changed = std::fabs(ox - prev_off.first) > 1e-6 || std::fabs(oy - prev_off.second) > 1e-6;
    if (ox != 0.0 || oy != 0.0) {
      off_[nd.id] = {ox, oy};
      pref_side_ = (ox * nx + oy * ny) > 0.0 ? 1 : -1;
    } else {
      off_.erase(nd.id);
    }
    px = nd.x + ox; py = nd.y + oy;
    out.push_back(pl);
  }
  return out;
}

}  // namespace scv_bt_planner
