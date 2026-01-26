SMPPI Controller 코드 분석

  이 코드는 Sampling-based Model Predictive Path Integral (SMPPI) 제어기로, PyTorch 기반의 병렬화된 궤적 최적화를 통해 자율주행 차량을
  제어합니다.

  ---
  📁 프로젝트 구조

  smppi/
  ├── scripts/
  │   ├── smppi_controller_node.py          # 메인 ROS2 노드
  │   ├── mppi_main_node.py                 # (대체 엔트리포인트)
  │   ├── sensor_processor_node.py          # 센서 전처리 노드
  │   ├── costmap_processor_node.py         # 코스트맵 처리
  │   └── visualization_node.py             # 시각화
  ├── smppi_controller/
  │   ├── optimizer/
  │   │   ├── smppi_optimizer.py           # 핵심 MPPI 최적화 로직
  │   │   └── trajectory_sampler.py        # 궤적 샘플링
  │   ├── critics/
  │   │   ├── goal_critic.py               # 목표 추적 비용
  │   │   └── obstacle_critic.py           # 장애물 회피 비용
  │   ├── motion_models/
  │   │   └── ackermann_model.py           # 차량 동역학 모델
  │   └── utils/
  │       ├── sensor_processor.py          # 센서 데이터 처리
  │       ├── transforms.py                # 좌표 변환
  │       └── geometry.py                  # 기하학 유틸리티
  └── config/
      └── smppi_params.yaml                # 파라미터 설정

  ---
  🎯 핵심 개념: MPPI 알고리즘

  **Model Predictive Path Integral (MPPI)**는 확률적 최적 제어 알고리즘입니다:

  1. 샘플링: K개의 제어 시퀀스를 노이즈와 함께 샘플링
  2. 시뮬레이션: 각 제어 시퀀스로 차량 동역학 롤아웃
  3. 비용 평가: 각 궤적의 비용 계산 (목표 추적 + 장애물 회피)
  4. 가중 평균: 낮은 비용의 궤적에 높은 가중치를 주어 제어 업데이트
  5. 반복: MPC 스타일로 첫 번째 제어만 적용하고 시퀀스 시프트

  ---
  🔥 메인 컨트롤러 노드 (smppi_controller_node.py)

  smppi_controller_node.py:43-669

  주요 토픽

  구독:
  - /ptl/scan - 레이저 스캔 (장애물)
  - /odom - 차량 상태 (위치, 속도)
  - /subgoal - 단일 목표점 (Behavior Planner로부터)
  - /multiple_waypoints - 다중 웨이포인트 (옵션)

  발행:
  - /ackermann_like_controller/cmd_vel - 제어 명령 (Twist)
  - /goal_status - 목표 도달 상태
  - /mppi_optimal_path - 최적 궤적 (시각화용)
  - /smppi_visualization - RViz 마커

  제어 루프 (control_callback)

  smppi_controller_node.py:336-412

  20Hz 주기로 실행되는 메인 제어 루프:

  1. is_ready() 체크 (로봇 상태, 장애물 있는지)
  2. optimizer.prepare() - 현재 상태 설정
  3. optimizer.set_obstacles() - 장애물 설정
  4. optimizer.optimize() - MPPI 최적화 수행
  5. optimizer.get_control_command() - 첫 번째 제어 추출
  6. cmd_pub.publish(cmd_vel) - 제어 명령 발행
  7. optimizer.shift_control_sequence() - MPC 시퀀스 시프트

  성능: 100회마다 계산 시간 로깅 (일반적으로 ~10-50ms)

  ---
  ⚙️ SMPPI Optimizer (smppi_optimizer.py)

  smppi_optimizer.py:25-504

  핵심 파라미터

  K = 3000              # 샘플 궤적 수 (배치 크기)
  T = 30                # 예측 시간 스텝 (3초 @ 0.1s)
  dt = 0.1              # 시뮬레이션 간격
  temperature = 1.8     # 소프트맥스 온도 (탐색 vs 이용)
  lambda_action = 0.08  # 제어 스무딩 가중치

  제어 표현: Derivative Control (U)

  중요한 설계 결정: 이 코드는 제어 미분(derivative control) 방식을 사용합니다.

  - U: 제어 변화량 시퀀스 [T, 2] (가속도, 조향각 변화율)
  - A: 실제 제어 시퀀스 [T, 2] (속도, 조향각)
  - 관계: A[t] = A[t-1] + U[t] * dt

  장점: 부드러운 제어 변화, MPC 시프트 시 안정성

  optimize() 메서드

  smppi_optimizer.py:107-173

  핵심 MPPI 알고리즘:

  1. 노이즈 샘플링: U_samples = U_nom + noise
  2. 제어 적분: A_samples = integrate_U_to_A(U_samples)
  3. 궤적 시뮬레이션: trajectories = simulate_from_A(A_samples)
  4. 비용 평가:
     - traj_costs = critics(trajectories)
     - action_costs = omega_cost(A_samples)
     - total_costs = traj_costs + lambda * action_costs
  5. 중요도 가중치:
     weights = exp(-(costs - min) / temperature)
     weights = weights / sum(weights)
  6. 제어 업데이트:
     dU = sum(weights * noise)
     U = U + dU

  디버그 메트릭: 엔트로피, 비용 평균/최소, 클램핑 비율 등 추적

  get_control_command() 메서드

  smppi_optimizer.py:300-399

  제어 추출 및 안전성 보장:

  1. 적분: A = integrate_U_to_A(a0, U) - 첫 번째 제어 계산
  2. 속도 클램핑: v ∈ [v_min, v_max], δ ∈ [w_min, w_max]
  3. 동적 조향각 제한 (NEW):
  δ_dyn = min(δ_max, atan(L * ay_max / v²))
    - 속도가 빠를수록 조향각 제한이 작아짐 (측방향 가속도 제한)
  4. 조향각 → 각속도 변환:
  ω = (v / L) * tan(δ)
  5. 각속도 캡핑:
  ω_cap = min((v/L)*tan(δ_max), ay_max/v)
  ω = clamp(ω, -ω_cap, ω_cap)
  6. Twist 발행

  중요: last_cmd_applied를 [v, δ]로 저장하여 다음 샘플링의 a0으로 사용

  ---
  🎯 Goal Critic (goal_critic.py)

  goal_critic.py:8-576

  목표 추적 전략

  Lookahead 기반 추적:
  - 차량 전방의 lookahead point를 계산
  - 궤적이 lookahead point를 지나가도록 유도
  - Pure Pursuit과 유사하지만 MPC 방식

  Lookahead 거리 계산

  goal_critic.py:234-268

  공식:
  lookahead = clamp(
      base_distance + velocity_factor * |v|,
      min_distance,
      max_distance
  )

  파라미터:
  - base_distance = 0.1m (기본 거리)
  - velocity_factor = 1.3s (속도 비례)
  - min_distance = 0.1m, max_distance = 7.0m

  예: v=2m/s → lookahead = 0.1 + 1.3*2 = 2.7m

  다중 웨이포인트 지원

  goal_critic.py:299-415

  복잡한 로직:
  1. 현재 목표까지 거리 < lookahead → 다음 웨이포인트로 연장
  2. 행동 그룹 변화 감지: 노드 타입이 다른 그룹으로 변경되면 연장 중단
    - Group 1: 후진 (2, 4)
    - Group 2: 전진 (1, 3, 5, 6, 9)
    - Group 3: 정지/신호등 (7, 8, 10)
  3. 커브 감지: 경로의 각도 변화가 25도 이상이면 lookahead 감소

  목적: 행동 변화 지점(전진→후진 등)에서 lookahead가 넘어가지 않도록 방지

  비용 함수 (간소화됨)

  goal_critic.py:71-232

  현재 활성화된 비용 (SIMPLIFIED VERSION):
  distance_cost = weight * sum(relu(distance - xy_tol) * time_weights)

  비활성화된 비용 (주석 처리됨):
  - heading_cost (heading 추적 비활성화 - 진동 방지)
  - alignment_cost (경로 정렬 비활성화)

  이유: 순수 거리 추적이 더 안정적이고 진동이 적음

  ---
  🚧 Obstacle Critic (obstacle_critic.py)

  obstacle_critic.py:15-311

  두 가지 모드

  1. Costmap 모드 (기본, 선호):
    - 그리드 기반 충돌 감지
    - O(1) lookup per point
    - 코스트맵 값 [0-100]을 비용으로 변환
  2. Point 모드 (레거시):
    - LaserScan 포인트 직접 사용
    - 원형 또는 다각형 충돌 감지

  Costmap 기반 비용 계산

  obstacle_critic.py:220-300

  코스트맵 값 → 비용 변환:
  0-49:   자유 공간 → 비용 0
  50-79:  inflation zone → repulsion_cost (거리 기반 척력)
  80-100: 점유 공간 → collision_cost (1000.0)

  병렬화:
  - 모든 [K, T+1] 포인트를 평탄화
  - 벡터화된 그리드 변환
  - NumPy 인덱싱으로 O(1) lookup
  - 궤적별 합산

  ---
  🚗 Ackermann Model (ackermann_model.py)

  ackermann_model.py:12-208

  차량 동역학

  입력: [v, δ] - 속도, 조향각출력: [x, y, θ] - 위치, 방향

  운동학 공식:
  ω = v * tan(δ) / L  # 각속도
  x_next = x + v * cos(θ) * dt
  y_next = y + v * sin(θ) * dt
  θ_next = θ + ω * dt

  배치 롤아웃

  ackermann_model.py:174-208

  효율적인 병렬 시뮬레이션:
  rollout_batch(initial_states, controls, dt):
      # initial_states: [K, 3]
      # controls: [K, T, 2]
      # 출력: trajectories [K, T+1, 3]

      for t in range(T):
          controls_t = validate_controls(controls[:, t, :])
          states = forward(states, controls_t, dt)

  K=3000개 궤적을 T=30 스텝씩 한 번에 시뮬레이션 → GPU 가속 가능

  ---
  📡 Sensor Processor (sensor_processor.py)

  sensor_processor.py:26-410

  LaserScan 처리 파이프라인

  sensor_processor.py:69-140

  1. 유효성 필터링:
     - 범위 체크 [0.1m, 5.0m]
     - 각도 필터링 (전방만)
     - NaN/Inf 제거

  2. 다운샘플링 (옵션):
     - downsample_factor로 포인트 수 감소

  3. 좌표 변환:
     - Polar (r, θ) → Cartesian (x, y) in laser frame
     - TF2로 laser → odom 프레임 변환

  4. Footprint 필터링 (옵션):
     - 차량 footprint 내부 점들 제거
     - 센서가 차체를 감지하는 것 방지

  5. 제한:
     - max_obstacles (1000개)로 제한

  TF2 좌표 변환

  sensor_processor.py:297-351

  장애물을 odom 프레임으로 변환:
  - Optimizer는 odom 프레임에서 작동
  - 레이저 스캔은 base_link 프레임
  - TF2로 실시간 변환

  ---
  📊 주요 파라미터 (smppi_params.yaml)

  optimizer:
    batch_size: 3000                    # K (샘플 수)
    time_steps: 30                      # T (예측 스텝)
    model_dt: 0.1                       # dt (시간 간격)
    temperature: 1.8                    # 탐색 정도
    lambda_action: 0.08                 # 스무딩 가중치
    noise_std_u: [0.40, 0.18]          # 노이즈 표준편차

  vehicle:
    wheelbase: 0.65                     # L (휠베이스)
    max_linear_velocity: 2.0            # v_max
    max_angular_velocity: 0.8           # ω_max
    max_steering_angle: 0.3665          # δ_max (21도)

  costs:
    obstacle_weight: 100.0              # 장애물 회피
    goal_weight: 30.0                   # 목표 추적

    lookahead:
      base_distance: 0.1
      velocity_factor: 1.3
      min_distance: 0.1
      max_distance: 7.0

  ---
  🔄 전체 데이터 흐름

  센서 입력:
    /ptl/scan → sensor_processor → ProcessedObstacles (odom frame)
    /odom → sensor_processor → robot_state [x,y,θ,v,ω]
    /subgoal → Transforms → goal_state [x,y,θ]

  메인 제어 루프 (20Hz):
    1. optimizer.prepare(robot_state, goal)
    2. optimizer.set_obstacles(obstacles)
    3. control_sequence = optimizer.optimize():
       a. U_samples = U_nom + noise         [K,T,2]
       b. A_samples = integrate(U_samples)  [K,T,2]
       c. trajectories = ackermann_model.rollout(A_samples) [K,T+1,3]
       d. goal_costs = goal_critic(trajectories)
       e. obs_costs = obstacle_critic(trajectories)
       f. total_costs = goal_costs + obs_costs + lambda*action_costs
       g. weights = softmax(-costs / temp)
       h. dU = weighted_sum(noise)
       i. U = U + dU
    4. cmd_vel = get_control_command(U[0])
    5. publish(cmd_vel)
    6. shift_control_sequence()

  출력:
    /ackermann_like_controller/cmd_vel → 차량 제어기
    /goal_status → Behavior Planner
    /mppi_optimal_path → RViz 시각화

  ---
  💡 주요 설계 특징

  1. Derivative Control (U 표현)

  - 제어 변화량을 최적화 변수로 사용
  - 부드러운 제어 변화 보장
  - MPC 시프트 시 안정성 향상

  2. 속도 적응형 제약

  - 조향각 제한: 고속에서 더 작은 δ 허용 (측방향 가속도 제한)
  - Lookahead 거리: 속도에 비례하여 증가 (고속에서 더 먼 전방 주시)

  3. 행동 인식 Lookahead

  - 노드 타입 변화 감지 (전진→후진 등)
  - 행동 경계에서 lookahead 연장 중단
  - 커브 감지하여 lookahead 감소

  4. 병렬화 최적화

  - PyTorch로 GPU 가속 가능
  - 배치 연산으로 K=3000 궤적 동시 처리
  - 벡터화된 비용 계산

  5. 안정성 우선

  - Heading 추적 비활성화 (진동 방지)
  - 동적 제약 적용 (물리적 한계 존중)
  - 다중 안전장치 (클램핑, 캡핑, 검증)

  ---
  🔗 Behavior Planner와의 통합

  Behavior Planner → SMPPI:
  - /multiple_waypoints: 현재 목표 + 다음 3개 웨이포인트
    - current_goal_node_type: 행동 타입
  - /mppi_update_params: 동적 파라미터 업데이트
    - max_linear_velocity, min_linear_velocity
    - respect_reverse_heading (후진 모드)
    - goal_weight, obstacle_weight
  - /pause_command: 일시정지 명령 (미구현)

  SMPPI → Behavior Planner:
  - /goal_status: 목표 도달 상태
    - distance_to_goal
    - goal_reached (threshold 1.5m)
    - goal_id

  ---
  📈 성능 특성

  - 제어 주파수: 20Hz (50ms 주기)
  - 최적화 시간: ~10-50ms (K=3000, T=30)
  - 예측 지평: 3초 (30 스텝 × 0.1s)
  - 샘플 수: 3000개 궤적
  - 지연 시간: ~50ms (센서 → 제어 명령)

  이 시스템은 실시간 확률적 최적 제어를 통해 복잡한 환경에서 부드럽고 안전한 주행을 실현합니다.