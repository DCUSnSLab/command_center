# Map Generation Tools

## bag_to_map_generator.py

ROS2 bag 파일에서 GPS 데이터를 읽어와서 3x3_map.json과 동일한 형식의 맵 파일을 생성합니다.

### 설치

```bash
# 의존성 설치
pip install pyproj

# 패키지 빌드
cd /path/to/workspace
colcon build --packages-select gmserver
source install/setup.bash
```

### 사용법

```bash
# ros2 run으로 실행
ros2 run gmserver bag_to_map_generator.py <bag_path> <gps_topic> [-o output_file] [-d distance]

# 예시
ros2 run gmserver bag_to_map_generator.py /path/to/rosbag2_folder /gps/fix -o my_map.json -d 1.0

# 직접 실행
python3 bag_to_map_generator.py /path/to/rosbag2_folder /gps/fix -o my_map.json -d 1.0
```

### 매개변수

- `bag_path`: ROS2 bag 파일이 있는 디렉토리 경로
- `gps_topic`: GPS 토픽 이름 (예: /gps/fix, /ublox_gps/fix)
- `-o, --output`: 출력 JSON 파일 경로 (기본값: generated_map.json)
- `-d, --distance`: GPS 포인트 간 최소 거리(미터) (기본값: 1.0)

### 지원하는 GPS 메시지 타입

- `sensor_msgs/NavSatFix`
- `geometry_msgs/PoseStamped`
- latitude/longitude 또는 lat/lon 필드를 가진 커스텀 메시지

### 출력 형식

생성되는 JSON 파일은 3x3_map.json과 동일한 구조를 가집니다:
- Node: GPS 포인트별로 생성되는 노드
- Link: 연속된 노드들을 연결하는 링크
- GPS 좌표와 UTM 좌표 모두 포함

## csv_to_map_generator.py

CSV 파일에서 GPS 데이터를 읽어와서 3x3_map.json과 동일한 형식의 맵 파일을 생성합니다.

### 설치

```bash
# 의존성 설치
pip install pyproj

# 패키지 빌드
cd /path/to/workspace
colcon build --packages-select gmserver
source install/setup.bash
```

### 사용법

```bash
# ros2 run으로 실행
ros2 run gmserver csv_to_map_generator.py <csv_path> [-o output_file] [-d distance] [--filter-option N]

# 예시
ros2 run gmserver csv_to_map_generator.py csv_map/mando.csv -o mando_map.json -d 2.0

# 직접 실행
python3 csv_to_map_generator.py csv_map/mando.csv -o mando_map.json -d 2.0

# 특정 옵션값으로 필터링 (예: 특수 구간만)
python3 csv_to_map_generator.py csv_map/mando.csv -o special_map.json --filter-option 10

# 통계만 보기
python3 csv_to_map_generator.py csv_map/mando.csv --stats-only
```

### 매개변수

- `csv_path`: GPS 데이터가 있는 CSV 파일 경로
- `-o, --output`: 출력 JSON 파일 경로 (기본값: csv_generated_map.json)
- `-d, --distance`: GPS 포인트 간 최소 거리(미터) (기본값: 1.0)
- `--filter-option`: 특정 옵션값으로 필터링 (예: 0, 10)
- `--stats-only`: 맵 생성 없이 통계만 출력

### 지원하는 CSV 형식

```csv
seq,llatitude,longitude,llatitude_utm,longitude_utm,option
135396,37.23965631,126.7736361,302516.9182,4123781.294,0
135417,37.23965414,126.7736412,302517.3636,4123781.043,0
```

대체 컬럼명도 지원:
- `lat`, `latitude` → `llatitude`
- `lon`, `lng` → `longitude`

### 출력 형식

- **Node**: CSV의 각 GPS 포인트에서 생성
- **Link**: 연속된 노드들을 연결, option 값에 따른 링크 타입 분류
- **Statistics**: 생성된 맵의 통계 정보
- **ExtraInfo**: 원본 CSV의 seq, option 정보 보존