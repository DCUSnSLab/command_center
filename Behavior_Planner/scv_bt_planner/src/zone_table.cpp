#include "scv_bt_planner/zone_table.hpp"

#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace scv_bt_planner {

namespace {
const char* KNOWN[] = {"sidewalk", "road", "crosswalk", "shared_road", "gps_denied"};
}

bool ZoneTable::isKnownZone(const std::string& z) const
{
  for (const char* k : KNOWN) if (z == k) return true;
  return false;
}

bool ZoneTable::loadFromFile(const std::string& path, std::string* error)
{
  std::ifstream f(path);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return loadFromString(ss.str(), error);
}

bool ZoneTable::loadFromString(const std::string& json_text, std::string* error)
{
  zones_.clear();
  node_count_ = 0;
  nlohmann::json g;
  try {
    g = nlohmann::json::parse(json_text);
  } catch (const std::exception& e) {
    if (error) *error = std::string("json parse: ") + e.what();
    return false;
  }
  // flat 스키마("Node") 와 소문자("nodes") 둘 다 허용
  const nlohmann::json* nodes = nullptr;
  if (g.contains("Node") && g["Node"].is_array()) nodes = &g["Node"];
  else if (g.contains("nodes") && g["nodes"].is_array()) nodes = &g["nodes"];
  if (!nodes) {
    if (error) *error = "no Node/nodes array";
    return false;
  }
  for (const auto& n : *nodes) {
    ++node_count_;
    std::string id;
    if (n.contains("ID") && n["ID"].is_string()) id = n["ID"].get<std::string>();
    else if (n.contains("id") && n["id"].is_string()) id = n["id"].get<std::string>();
    if (id.empty()) continue;
    std::string zone;
    if (n.contains("Zone") && n["Zone"].is_string()) zone = n["Zone"].get<std::string>();
    else if (n.contains("zone") && n["zone"].is_string()) zone = n["zone"].get<std::string>();
    if (!zone.empty()) zones_[id] = zone;
  }
  return true;
}

std::string ZoneTable::zoneOf(const std::string& node_id) const
{
  auto it = zones_.find(node_id);
  if (it == zones_.end()) return DEFAULT_ZONE;
  return it->second;
}

}  // namespace scv_bt_planner
