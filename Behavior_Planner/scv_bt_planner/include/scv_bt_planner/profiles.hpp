// 행동 프로필: smppi baseline × node_type 수정자 × BT 오버레이 → 제어기 파라미터.
//
// simple_behavior_planner/behavior_parameter_manager.py 와 같은 규칙:
//   - baseline 은 smppi_params.yaml 의 정해진 키만 뽑는다(원본과 같은 키·기본값)
//   - multipliers 는 baseline 에 **존재하는 키에만** 곱한다 (없는 키는 무시)
//   - overrides 는 절대값으로 덮어쓴다 (bool 은 0/1)
// 그 위에 BT 오버레이(구역·회전)를 한 번 더 곱한다 — 오버레이가 비면 원본과 동일.
#pragma once

#include <map>
#include <optional>
#include <string>

namespace scv_bt_planner {

using ParamMap = std::map<std::string, double>;

struct ProfileDef {
  std::string name;
  std::string description;
  int code = 0;             // MPPIParams.current_behavior_type 에 실을 값 (node 프로필은 node_type)
  ParamMap multipliers;
  ParamMap overrides;
};

class Profiles {
public:
  static ParamMap defaultBaseline();

  // profiles yaml (node_profiles / overlays / smppi_config_path)
  bool loadProfiles(const std::string& yaml_path, std::string* error = nullptr);
  // smppi_params.yaml → baseline. 실패하면 defaultBaseline() 을 쓰고 false.
  bool loadBaseline(const std::string& smppi_yaml_path, std::string* error = nullptr);

  const std::string& smppiConfigPath() const { return smppi_config_path_; }
  const ParamMap& baseline() const { return baseline_; }
  bool hasNodeProfile(int node_type) const { return node_profiles_.count(node_type) > 0; }
  const ProfileDef* nodeProfile(int node_type) const;
  const ProfileDef* overlay(const std::string& name) const;
  size_t nodeProfileCount() const { return node_profiles_.size(); }
  size_t overlayCount() const { return overlays_.size(); }

  // node_type 프로필(미지 유형은 1 로 폴백) 위에 overlay(빈 문자열 = 없음) 적용.
  // description/code 는 MPPIParams 메타데이터용.
  ParamMap compute(int node_type, const std::string& overlay_name,
                   std::string* description = nullptr, int* code = nullptr) const;

  // 원본 validate_behavior_params 와 동일
  static bool validate(const ParamMap& p, std::string* why = nullptr);

private:
  std::string smppi_config_path_ = "package://smppi/config/smppi_params.yaml";
  ParamMap baseline_ = defaultBaseline();
  std::map<int, ProfileDef> node_profiles_;
  std::map<std::string, ProfileDef> overlays_;
};

}  // namespace scv_bt_planner
