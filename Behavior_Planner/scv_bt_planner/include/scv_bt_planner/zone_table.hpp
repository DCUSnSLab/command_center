// graph.json 의 노드별 구역(Zone) 속성 조회.
//
// 지도 스키마 확장(2026-09-28 설계): Node 에 "Zone" 문자열을 추가한다.
//   sidewalk | road | crosswalk | shared_road | gps_denied
// 필드가 없으면 sidewalk 로 간주하므로 기존 지도는 그대로 동작한다.
// MapNode 메시지는 바꾸지 않고, BT 플래너가 같은 graph.json 을 직접 읽어
// PlannedPath 의 노드 ID 로 조회한다.
#pragma once

#include <string>
#include <unordered_map>

namespace scv_bt_planner {

class ZoneTable {
public:
  static constexpr const char* DEFAULT_ZONE = "sidewalk";

  // graph.json 로드. 실패 시 false (테이블은 비어 있고 zoneOf 는 기본값을 돌려준다).
  bool loadFromFile(const std::string& path, std::string* error = nullptr);
  bool loadFromString(const std::string& json_text, std::string* error = nullptr);

  std::string zoneOf(const std::string& node_id) const;
  bool has(const std::string& node_id) const { return zones_.count(node_id) > 0; }
  size_t size() const { return zones_.size(); }
  size_t nodeCount() const { return node_count_; }
  bool isKnownZone(const std::string& z) const;

private:
  std::unordered_map<std::string, std::string> zones_;  // 명시된 노드만
  size_t node_count_ = 0;
};

}  // namespace scv_bt_planner
