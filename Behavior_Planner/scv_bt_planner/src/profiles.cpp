#include "scv_bt_planner/profiles.hpp"

#include <yaml-cpp/yaml.h>

namespace scv_bt_planner {

namespace {

double toDouble(const YAML::Node& n, double dflt)
{
  if (!n || !n.IsScalar()) return dflt;
  // bool → 0/1, 숫자 → 값
  try {
    const std::string s = n.as<std::string>();
    if (s == "true" || s == "True") return 1.0;
    if (s == "false" || s == "False") return 0.0;
    return n.as<double>();
  } catch (...) {
    return dflt;
  }
}

void readMap(const YAML::Node& n, ParamMap& out)
{
  if (!n || !n.IsMap()) return;
  for (const auto& kv : n) {
    out[kv.first.as<std::string>()] = toDouble(kv.second, 0.0);
  }
}

ProfileDef readProfile(const YAML::Node& n, const std::string& fallback_name)
{
  ProfileDef p;
  p.name = n["name"] ? n["name"].as<std::string>() : fallback_name;
  p.description = n["description"] ? n["description"].as<std::string>() : p.name;
  p.code = n["code"] ? n["code"].as<int>() : 0;
  readMap(n["multipliers"], p.multipliers);
  readMap(n["overrides"], p.overrides);
  return p;
}

}  // namespace

ParamMap Profiles::defaultBaseline()
{
  // behavior_parameter_manager._get_default_baseline() 과 동일
  return {
    {"max_linear_velocity", 3.0}, {"min_linear_velocity", 0.0},
    {"max_angular_velocity", 1.16}, {"min_angular_velocity", -1.16},
    {"wheelbase", 0.65}, {"max_steering_angle", 0.3665}, {"radius", 0.6},
    {"footprint_padding", 0.15}, {"obstacle_weight", 100.0}, {"goal_weight", 30.0},
    {"lookahead_base_distance", 1.0}, {"lookahead_velocity_factor", 0.4},
    {"lookahead_min_distance", 1.0}, {"lookahead_max_distance", 6.0},
    {"goal_reached_threshold", 2.0}, {"control_frequency", 20.0},
    {"xy_goal_tolerance", 2.0}, {"yaw_goal_tolerance", 0.25}};
}

bool Profiles::loadProfiles(const std::string& yaml_path, std::string* error)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(yaml_path);
  } catch (const std::exception& e) {
    if (error) *error = std::string("profiles yaml: ") + e.what();
    return false;
  }
  if (root["smppi_config_path"]) smppi_config_path_ = root["smppi_config_path"].as<std::string>();
  node_profiles_.clear();
  overlays_.clear();
  if (root["node_profiles"] && root["node_profiles"].IsMap()) {
    for (const auto& kv : root["node_profiles"]) {
      const int t = kv.first.as<int>();
      ProfileDef p = readProfile(kv.second, "type_" + std::to_string(t));
      if (p.code == 0) p.code = t;
      node_profiles_[t] = p;
    }
  }
  if (root["overlays"] && root["overlays"].IsMap()) {
    for (const auto& kv : root["overlays"]) {
      const std::string name = kv.first.as<std::string>();
      ProfileDef p = readProfile(kv.second, name);
      p.name = name;
      overlays_[name] = p;
    }
  }
  if (node_profiles_.empty()) {
    if (error) *error = "profiles yaml has no node_profiles";
    return false;
  }
  return true;
}

bool Profiles::loadBaseline(const std::string& smppi_yaml_path, std::string* error)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(smppi_yaml_path);
  } catch (const std::exception& e) {
    baseline_ = defaultBaseline();
    if (error) *error = std::string("smppi yaml: ") + e.what();
    return false;
  }
  YAML::Node rp = root["/**"] ? root["/**"]["ros__parameters"] : YAML::Node();
  if (!rp) {
    baseline_ = defaultBaseline();
    if (error) *error = "smppi yaml: no /**/ros__parameters";
    return false;
  }
  // 원본 _load_smppi_baseline 과 같은 키·기본값. xy/yaw_goal_tolerance 는 **뽑지 않는다**
  // (원본도 파일 로드 경로에서는 넣지 않으므로 그 키의 multiplier 는 무시된다).
  ParamMap b;
  YAML::Node v = rp["vehicle"];
  b["max_linear_velocity"] = toDouble(v["max_linear_velocity"], 3.0);
  b["min_linear_velocity"] = toDouble(v["min_linear_velocity"], 0.0);
  b["max_angular_velocity"] = toDouble(v["max_angular_velocity"], 1.16);
  b["min_angular_velocity"] = toDouble(v["min_angular_velocity"], -1.16);
  b["wheelbase"] = toDouble(v["wheelbase"], 0.65);
  b["max_steering_angle"] = toDouble(v["max_steering_angle"], 0.3665);
  b["radius"] = toDouble(v["radius"], 0.6);
  b["footprint_padding"] = toDouble(v["footprint_padding"], 0.15);
  YAML::Node c = rp["costs"];
  b["obstacle_weight"] = toDouble(c["obstacle_weight"], 100.0);
  b["goal_weight"] = toDouble(c["goal_weight"], 30.0);
  YAML::Node la = c["lookahead"];
  b["lookahead_base_distance"] = toDouble(la["base_distance"], 1.0);
  b["lookahead_velocity_factor"] = toDouble(la["velocity_factor"], 0.4);
  b["lookahead_min_distance"] = toDouble(la["min_distance"], 1.0);
  b["lookahead_max_distance"] = toDouble(la["max_distance"], 6.0);
  b["goal_reached_threshold"] = toDouble(rp["goal_reached_threshold"], 2.0);
  b["control_frequency"] = toDouble(rp["control_frequency"], 20.0);
  YAML::Node o = rp["optimizer"];
  b["batch_size"] = toDouble(o["batch_size"], 3000);
  b["time_steps"] = toDouble(o["time_steps"], 30);
  b["model_dt"] = toDouble(o["model_dt"], 0.1);
  b["temperature"] = toDouble(o["temperature"], 1.8);
  b["lambda_action"] = toDouble(o["lambda_action"], 0.08);
  baseline_ = b;
  return true;
}

const ProfileDef* Profiles::nodeProfile(int node_type) const
{
  auto it = node_profiles_.find(node_type);
  return it == node_profiles_.end() ? nullptr : &it->second;
}

const ProfileDef* Profiles::overlay(const std::string& name) const
{
  auto it = overlays_.find(name);
  return it == overlays_.end() ? nullptr : &it->second;
}

ParamMap Profiles::compute(int node_type, const std::string& overlay_name,
                           std::string* description, int* code) const
{
  const ProfileDef* np = nodeProfile(node_type);
  if (!np) { np = nodeProfile(1); node_type = 1; }   // 원본: 미지 유형은 1
  ParamMap p = baseline_;
  if (np) {
    for (const auto& [k, m] : np->multipliers) {
      auto it = baseline_.find(k);
      if (it != baseline_.end()) p[k] = it->second * m;
    }
    for (const auto& [k, v] : np->overrides) p[k] = v;
  }
  std::string desc = np ? np->description : ("Behavior " + std::to_string(node_type));
  int c = node_type;
  if (!overlay_name.empty()) {
    const ProfileDef* ov = overlay(overlay_name);
    if (ov) {
      for (const auto& [k, m] : ov->multipliers) {
        auto it = p.find(k);
        if (it != p.end()) it->second *= m;
      }
      for (const auto& [k, v] : ov->overrides) p[k] = v;
      desc = ov->description + " (" + desc + ")";
      if (ov->code != 0) c = ov->code;
    }
  }
  if (description) *description = desc;
  if (code) *code = c;
  return p;
}

bool Profiles::validate(const ParamMap& p, std::string* why)
{
  auto get = [&](const char* k, double d) { auto it = p.find(k); return it == p.end() ? d : it->second; };
  const double max_vel = get("max_linear_velocity", 0.0);
  const double min_vel = get("min_linear_velocity", 0.0);
  if (max_vel < 0 && min_vel >= 0) { if (why) *why = "invalid velocity config"; return false; }
  if (get("goal_weight", 1.0) <= 0) { if (why) *why = "invalid goal_weight"; return false; }
  if (get("lookahead_min_distance", 0.0) >= get("lookahead_max_distance", 1.0)) {
    if (why) *why = "invalid lookahead";
    return false;
  }
  return true;
}

}  // namespace scv_bt_planner
