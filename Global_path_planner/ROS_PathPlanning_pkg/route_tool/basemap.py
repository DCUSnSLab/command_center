"""위성 배경 — XYZ 타일을 받아 map(datum-local) 좌표에 붙인다.

SCV_MapManager/map_editor 에서 가져왔다. map_editor 는 ROS 를 넣지 않고 분리해
둘 예정이라 공유 라이브러리로 묶지 않고 복사한다. 순수 PyQt+urllib 라 의존성 없음.

왜 타일마다 변환을 따로 두는가:
  타일은 웹 메르카토르(EPSG:3857), 우리 좌표는 UTM datum-local 이다. 둘 다 등각이라
  작은 범위에서는 닮은꼴이지만 **자오선 수렴각만큼 기울어 있다**. 우리 datum
  (128.80°E, zone 52 중앙자오선 129°E)에서는 -0.115°, 600 m 범위 끝에서 1.2 m 어긋난다.
  타일마다 세 모서리를 실제로 투영해 아핀 변환을 만들면 회전·축척이 자동으로 맞는다.

정렬 주의:
  위성영상 자체의 지오레퍼런싱 오차가 수 m 이고, 우리 맵도 GPS 자기 일관성이 2.3 m 다.
  **배경은 위치 감을 잡는 보조 수단이지 정렬 기준이 아니다.** 위성에 맞춰 노드를
  옮기지 말 것.
"""
import math
import os
import threading
import urllib.request
from collections import OrderedDict

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

# Mapbox 키는 SCV_PathEditor 에 설정돼 있던 것을 그대로 쓴다 (계정 키).
_MAPBOX_KEY = ("pk.eyJ1IjoiYmFja2dyb3VuZG1pbiIsImEiOiJjbWZnZDNoZm0wMGQ1MmpxMHllYWJqYW5nIn0"
               ".OGk1et_KaOJONN7kZwOBeQ")

#: max_native = 그 소스가 실제로 갖고 있는 최대 줌. 넘겨 요청하면 빈 타일이 온다.
PROVIDERS = {
    "off": None,
    "google": {
        "label": "Google 위성",
        "url": "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        "subdomains": "0123",
        "max_native": 21,
        "tile_px": 256,
        "attribution": "© Google",
    },
    "mapbox": {
        "label": "Mapbox 위성",
        "url": ("https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png"
                "?access_token=" + _MAPBOX_KEY),
        "subdomains": "",
        "max_native": 22,
        "tile_px": 512,          # @2x — 타일이 덮는 범위는 같고 화소만 2배
        "attribution": "© Mapbox © Maxar",
    },
}
ORDER = ("off", "google", "mapbox")


# ---------------------------------------------------------------- 타일 좌표
def deg2tile(lat_deg, lon_deg, z):
    """위경도 -> 타일 좌표(실수). 정수부가 타일 번호, 소수부가 타일 안 위치."""
    lat = math.radians(max(min(lat_deg, 85.05112878), -85.05112878))
    n = 2.0 ** z
    return ((lon_deg + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(lat)) / math.pi) / 2.0 * n)


def tile2deg(xt, yt, z):
    """타일 좌표(실수) -> 위경도. 정수를 주면 그 타일의 북서 모서리."""
    n = 2.0 ** z
    lon = xt / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yt / n))))
    return lat, lon


def pick_zoom(scale_px_per_m, lat_deg, provider):
    """화면 축척에 맞는 줌. 타일 1픽셀 ≈ 화면 1픽셀이 되게 고른다."""
    p = PROVIDERS.get(provider)
    if not p:
        return 0
    # 줌 z 에서 지상해상도 = 156543.03392 * cos(lat) / 2^z  (m/px, 256px 타일 기준)
    z = math.log2(156543.03392 * math.cos(math.radians(lat_deg)) * max(scale_px_per_m, 1e-6))
    return max(0, min(p["max_native"], int(round(z))))


# ---------------------------------------------------------------- 캐시·로더
class _Cache:
    """메모리 LRU + 디스크. 디스크는 세션이 바뀌어도 남아 오프라인에서도 쓴다."""

    def __init__(self, root=None, mem_max=512):
        self.root = root or os.path.expanduser("~/.cache/scv_map_editor/tiles")
        self.mem = OrderedDict()
        self.mem_max = mem_max
        self.lock = threading.Lock()

    def _path(self, provider, z, x, y):
        return os.path.join(self.root, provider, str(z), str(x), f"{y}.png")

    def get(self, provider, z, x, y):
        key = (provider, z, x, y)
        with self.lock:
            img = self.mem.get(key)
            if img is not None:
                self.mem.move_to_end(key)
                return img
        fp = self._path(provider, z, x, y)
        if os.path.isfile(fp):
            img = QImage(fp)
            if not img.isNull():
                self.put_mem(key, img)
                return img
        return None

    def put_mem(self, key, img):
        with self.lock:
            self.mem[key] = img
            self.mem.move_to_end(key)
            while len(self.mem) > self.mem_max:
                self.mem.popitem(last=False)

    def put_disk(self, provider, z, x, y, data):
        fp = self._path(provider, z, x, y)
        try:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            tmp = fp + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, fp)
        except OSError:
            pass


class BasemapLoader(QObject):
    """타일을 백그라운드로 받아온다. 받으면 tileReady 로 다시 그리라고 알린다.

    GUI 스레드에서 네트워크를 만지면 안 되므로 워커 스레드에서만 받고, 화면 갱신은
    신호로 넘긴다. 같은 타일을 두 번 요청하지 않도록 진행 중 목록을 둔다.
    """

    tileReady = pyqtSignal()

    def __init__(self, parent=None, workers=4):
        super().__init__(parent)
        self.cache = _Cache()
        self._inflight = set()
        self._queue = []
        self._lock = threading.Lock()
        self._threads = []
        self._stop = False
        self._wake = threading.Semaphore(0)
        for _ in range(workers):
            t = threading.Thread(target=self._work, daemon=True)
            t.start()
            self._threads.append(t)

    def get(self, provider, z, x, y):
        """캐시에 있으면 즉시 반환, 없으면 None 을 주고 백그라운드로 받아온다."""
        img = self.cache.get(provider, z, x, y)
        if img is not None:
            return img
        key = (provider, z, x, y)
        with self._lock:
            if key in self._inflight:
                return None
            self._inflight.add(key)
            self._queue.append(key)
        self._wake.release()
        return None

    def _url(self, provider, z, x, y):
        p = PROVIDERS[provider]
        u = p["url"].replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
        if "{s}" in u:
            sub = p["subdomains"]
            u = u.replace("{s}", sub[(x + y) % len(sub)])
        return u

    def _work(self):
        while not self._stop:
            self._wake.acquire()
            if self._stop:
                return
            with self._lock:
                if not self._queue:
                    continue
                key = self._queue.pop()          # 최신 요청부터 (화면에 보이는 것)
            provider, z, x, y = key
            try:
                req = urllib.request.Request(
                    self._url(provider, z, x, y),
                    # UA 가 없으면 403 을 주는 소스가 있다
                    headers={"User-Agent": "Mozilla/5.0 (SCV Map Editor)"})
                with urllib.request.urlopen(req, timeout=8) as r:
                    data = r.read()
                img = QImage()
                if img.loadFromData(data) and not img.isNull():
                    self.cache.put_mem(key, img)
                    self.cache.put_disk(provider, z, x, y, data)
                    self.tileReady.emit()
            except Exception:
                pass                              # 못 받은 타일은 그냥 빈칸으로 둔다
            finally:
                with self._lock:
                    self._inflight.discard(key)

    def stop(self):
        self._stop = True
        for _ in self._threads:
            self._wake.release()


# ---------------------------------------------------------------- 좌표 변환
class LocalProjector:
    """번들 datum 기준 위경도 <-> map-local(m) 변환. 결과를 캐싱해 반복 호출을 줄인다."""

    def __init__(self, bundle):
        self.b = bundle
        u = bundle.utm or {}
        self.zone = int(u.get("zone", 52))
        self.oe = float(u.get("origin_easting", 0.0))
        self.on = float(u.get("origin_northing", 0.0))
        self.northern = bool(u.get("northern", True))

    def to_local(self, lat, lon):
        import utm as U
        e, n, _, _ = U.from_latlon(lat, lon, force_zone_number=self.zone)
        return e - self.oe, n - self.on

    def to_latlon(self, x, y):
        return self.b.latlon_of(x, y)
