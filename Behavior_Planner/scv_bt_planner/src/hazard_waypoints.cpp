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

double HazardWaypoints::maxLateral(double d, double R)
{
  if (R <= 0.0) return 1e9;
  if (d <= 0.0) return 0.0;
  if (d >= 2.0 * R) return 1e9;
  const double th = std::asin(d / (2.0 * R));
  return 2.0 * R * (1.0 - std::cos(th));
}

std::vector<HazardPlacement> HazardWaypoints::plan(const std::vector<PathNode>& upcoming, const PathNode* prev_node,
                                                   double sx, double sy, const BlockedAt& blocked)
{
  const size_t n = std::min(upcoming.size(), static_cast<size_t>(std::max(0, p_.lookahead) + 1));
  std::vector<HazardPlacement> out(n);
  if (n == 0) return out;

  // 노드별 진행 방향·왼쪽 법선
  std::vector<double> ux(n), uy(n), nx(n), ny(n);
  for (size_t i = 0; i < n; ++i) {
    const PathNode& nd = upcoming[i];
    double tx, ty;
    if (i + 1 < upcoming.size()) { tx = upcoming[i + 1].x - nd.x; ty = upcoming[i + 1].y - nd.y; }
    else if (i > 0) { tx = nd.x - upcoming[i - 1].x; ty = nd.y - upcoming[i - 1].y; }
    else if (prev_node) { tx = nd.x - prev_node->x; ty = nd.y - prev_node->y; }
    else { tx = nd.x - sx; ty = nd.y - sy; }
    const double tl = std::hypot(tx, ty);
    ux[i] = tl > 1e-6 ? tx / tl : 1.0; uy[i] = tl > 1e-6 ? ty / tl : 0.0;
    nx[i] = -uy[i]; ny[i] = ux[i];
  }
  auto at = [&](size_t i, double lat) {
    return std::make_pair(upcoming[i].x + lat * nx[i], upcoming[i].y + lat * ny[i]);
  };
  auto free_at = [&](size_t i, double lat) { auto q = at(i, lat); return pointFree(q.first, q.second, blocked); };
  auto budget = [&](double along) { return p_.feasible_factor * maxLateral(along, p_.turn_radius); };

  // 1) 노드별 필요한 횡위치(기구학 무시). 판정 범위 밖은 기존 값 유지.
  std::vector<double> lat(n, 0.0);
  std::vector<bool> eval(n, false), nofree(n, false), fixed(n, false);   // fixed: 위험 때문에 정한(또는 유지 중인) 횡위치
  for (size_t i = 0; i < n; ++i) {
    const PathNode& nd = upcoming[i];
    out[i].id = nd.id;
    const auto po = offset(nd.id);
    const double plat = po.first * nx[i] + po.second * ny[i];
    const double dveh = std::hypot(nd.x - sx, nd.y - sy);
    if (dveh < p_.min_dist || dveh > p_.max_dist) {
      lat[i] = plat; out[i].out_of_range = true;
      continue;
    }
    eval[i] = true;
    if (plat != 0.0 && free_at(i, plat)) { lat[i] = plat; fixed[i] = true; continue; }   // 유지(히스테리시스)
    if (free_at(i, 0.0)) { lat[i] = 0.0; continue; }
    const int first = pref_side_ != 0 ? pref_side_ : -1;   // 미정이면 오른쪽 먼저(우측통행)
    bool found = false;
    for (double s = p_.step; s <= p_.max_shift + 1e-9 && !found; s += p_.step) {
      for (int side : {first, -first}) {
        if (free_at(i, side * s)) { lat[i] = side * s; found = true; fixed[i] = true; break; }
      }
    }
    if (!found) { nofree[i] = true; lat[i] = 0.0; }
  }
  // 이웃 노드 사이 횡변화 한도
  auto pair_budget = [&](size_t i, size_t k) {
    const double along = (upcoming[k].x - upcoming[i].x) * ux[k] + (upcoming[k].y - upcoming[i].y) * uy[k];
    return budget(along);
  };
  // 2) 뒤에서 앞으로: 다음 노드에 닿도록 앞 노드를 미리 당겨 놓는다(고정되지 않은 판정 대상만).
  for (size_t k = n - 1; k >= 1; --k) {
    const size_t i = k - 1;
    if (!eval[i] || fixed[i] || nofree[k]) continue;
    const double b = pair_budget(i, k);
    if (std::fabs(lat[k] - lat[i]) <= b + 1e-9) continue;
    const double want = lat[k] > lat[i] ? lat[k] - b : lat[k] + b;
    if (free_at(i, want)) lat[i] = want;
  }
  // 2') 앞에서 뒤로: 위험물을 지난 뒤 경로 중앙으로 서서히 돌아오게 다음 노드를 당긴다(고정되지 않은 판정 대상만).
  for (size_t k = 1; k < n; ++k) {
    const size_t i = k - 1;
    if (!eval[k] || fixed[k] || nofree[i]) continue;
    const double b = pair_budget(i, k);
    if (std::fabs(lat[k] - lat[i]) <= b + 1e-9) continue;
    const double want = lat[i] > lat[k] ? lat[i] - b : lat[i] + b;
    if (free_at(k, want)) lat[k] = want;
  }
  // 3) 차량에서 앞으로: 순서대로 닿을 수 있는가. 닿을 수 없거나 빈 곳이 없으면 skip 표시.
  double px = sx, py = sy;
  for (size_t i = 0; i < n; ++i) {
    auto& pl = out[i];
    const auto po = offset(pl.id);
    if (eval[i]) {
      const double prev_lat = (px - upcoming[i].x) * nx[i] + (py - upcoming[i].y) * ny[i];
      const double along = (upcoming[i].x - px) * ux[i] + (upcoming[i].y - py) * uy[i];
      if (nofree[i] || std::fabs(lat[i] - prev_lat) > budget(along) + 1e-9) {
        pl.skip = true;
        pl.changed = off_.erase(pl.id) > 0;
        continue;   // 직전 점 유지
      }
    }
    const auto q = at(i, lat[i]);
    pl.dx = q.first - upcoming[i].x; pl.dy = q.second - upcoming[i].y;
    if (eval[i]) {
      pl.changed = std::fabs(pl.dx - po.first) > 1e-6 || std::fabs(pl.dy - po.second) > 1e-6;
      if (lat[i] != 0.0) { off_[pl.id] = {pl.dx, pl.dy}; pref_side_ = lat[i] > 0.0 ? 1 : -1; }
      else off_.erase(pl.id);
    }
    px = q.first; py = q.second;
  }
  return out;
}

}  // namespace scv_bt_planner
