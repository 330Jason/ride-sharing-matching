"""即時共乘派單與最佳化配對系統 — 第五階段：即時車輛資料。

叫車平台（Uber、台灣大車隊等）的司機即時位置屬於業者私有資料，沒有公開 API，
爬取其 App 也違反服務條款與司機隱私。這裡改用臺北市資料大平臺公開、免金鑰的
「臺北市公車動態資訊」，以此刻真的在台北街道上行駛的公車作為司機位置。
- 座標慣例沿用：x 為經度 (lng)、y 為緯度 (lat)。
- 本模組不依賴地圖套件：「是否在路網上」的判斷函式由呼叫端注入，
  與 MatchingEngine 注入 cost_fn 的做法一致。
"""

from __future__ import annotations

import gzip
import json
import math
import random
import ssl
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Sequence, Tuple, TypeVar

import certifi

from matching_engine import Driver, Rider

# 全市公車的最新 GPS 位置（gzip 壓縮的 JSON，內容開頭帶 UTF-8 BOM）
BUS_DATA_URL = "https://tcgbusfs.blob.core.windows.net/blobbus/GetBusData.gz"
EARTH_RADIUS_M = 6_371_000
# Web Mercator：zoom 0 時赤道上每像素代表的公尺數（256px 圖磚）
METERS_PER_PX_AT_ZOOM_0 = 156_543.03392

T = TypeVar("T")


@dataclass
class BusPosition:
    """一輛公車的即時位置。"""

    bus_id: str          # 車牌
    x: float             # 經度 lng
    y: float             # 緯度 lat
    data_time: datetime  # 車機回報位置的時間（臺北當地時間）


def parse_bus_data(raw: bytes, max_age_s: int = 180) -> List[BusPosition]:
    """解析 GetBusData.gz 的原始內容，只保留此刻營運中、行車正常的公車。

    - DutyStatus == "1"（勤務中）且 BusStatus == "0"（行車正常）；
      其餘如 DutyStatus 0 = 車機剛開機尚未發車、2 = 已駛離末站，
      BusStatus 99 = 非營運狀態都略過。
    - 座標或時間無法解析、座標為 0 的紀錄略過。
    - 以資料中最新一筆的時間為基準，超過 max_age_s 秒沒回報位置的公車
      視為過時略過。以資料本身為基準，不受伺服器時區與時鐘影響。
    - 內容不是 gzip 或被截斷時拋 ValueError（gzip 原本會拋 OSError 或
      EOFError），讓呼叫端只需處理一種「格式錯誤」。
    """
    try:
        text = gzip.decompress(raw).decode("utf-8-sig")
    except (EOFError, OSError) as error:
        raise ValueError(f"公車資料不是完整的 gzip 檔：{error}") from error
    feed = json.loads(text)

    buses: List[BusPosition] = []
    for record in feed.get("BusInfo", []):
        if record.get("DutyStatus") != "1" or record.get("BusStatus") != "0":
            continue
        try:
            x = float(record["Longitude"])
            y = float(record["Latitude"])
            data_time = datetime.strptime(record["DataTime"], "%Y-%m-%d %H:%M:%S")
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)) or x == 0 or y == 0:
            continue
        buses.append(
            BusPosition(bus_id=str(record.get("BusID", "")), x=x, y=y, data_time=data_time)
        )

    if not buses:
        return []
    newest = max(bus.data_time for bus in buses)
    return [bus for bus in buses if (newest - bus.data_time).total_seconds() <= max_age_s]


def fetch_bus_positions(
    url: str = BUS_DATA_URL, timeout: float = 10.0
) -> List[BusPosition]:
    """下載並解析公車即時位置。

    明確以 certifi 的憑證清單驗證 HTTPS：python.org 的 macOS 版 Python
    預設讀不到系統憑證，urllib 會回報 CERTIFICATE_VERIFY_FAILED。
    網路斷線、逾時、HTTP 錯誤會拋 OSError（含 URLError），
    內容格式錯誤會拋 ValueError，由呼叫端決定如何提示使用者。
    """
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=timeout, context=context) as response:
        return parse_bus_data(response.read())


def haversine_m(x1: float, y1: float, x2: float, y2: float) -> float:
    """兩個 (lng, lat) 座標間的球面距離（公尺）。"""
    lat1, lat2 = math.radians(y1), math.radians(y2)
    d_lat = lat2 - lat1
    d_lng = math.radians(x2 - x1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def view_radius_m(
    center_lat: float,
    zoom: float,
    map_height_px: int,
    min_m: float = 300.0,
    max_m: float = 2_000.0,
) -> float:
    """依目前地圖縮放算出取樣半徑，讓取樣出的標記都落在畫面內。

    地圖寬大於高，所以以「半個畫面高度」為準，再乘 0.8 留邊；
    結果限制在 [min_m, max_m]：拉得很遠時不會取到整個台北市，
    放得很大時也至少保留幾百公尺可以取樣。
    """
    meters_per_px = METERS_PER_PX_AT_ZOOM_0 * math.cos(math.radians(center_lat)) / 2 ** zoom
    radius = map_height_px / 2 * meters_per_px * 0.8
    return min(max(radius, min_m), max_m)


def _shuffled_within(
    items: Sequence[T],
    coords: Callable[[T], Tuple[float, float]],
    center_x: float,
    center_y: float,
    radius_m: float,
    rng: random.Random,
) -> List[T]:
    """回傳距中心 radius_m 內的項目，順序以 rng 隨機打亂。"""
    nearby = [
        item for item in items
        if haversine_m(center_x, center_y, *coords(item)) <= radius_m
    ]
    rng.shuffle(nearby)
    return nearby


def pick_live_drivers(
    buses: Sequence[BusPosition],
    center_x: float,
    center_y: float,
    radius_m: float,
    count: int,
    rng: random.Random,
    is_on_network: Callable[[float, float], bool],
) -> Tuple[List[Driver], Dict[str, str]]:
    """從半徑內的公車隨機挑出最多 count 輛，轉成 Driver（id 依序為 D1, D2, ...）。

    is_on_network(x, y) 由呼叫端注入（例如以路網吸附距離判斷）；
    路網涵蓋範圍外的公車（如新北市）會被略過，且不佔名額。
    回傳 (drivers, 司機 id → 公車車牌)。
    """
    drivers: List[Driver] = []
    bus_ids: Dict[str, str] = {}
    candidates = _shuffled_within(
        buses, lambda bus: (bus.x, bus.y), center_x, center_y, radius_m, rng
    )
    for bus in candidates:
        if len(drivers) == count:
            break
        if not is_on_network(bus.x, bus.y):
            continue
        driver_id = f"D{len(drivers) + 1}"
        drivers.append(Driver(id=driver_id, x=bus.x, y=bus.y))
        bus_ids[driver_id] = bus.bus_id
    return drivers, bus_ids


def pick_random_riders(
    points: Sequence[Tuple[float, float]],
    center_x: float,
    center_y: float,
    radius_m: float,
    count: int,
    rng: random.Random,
) -> List[Rider]:
    """從半徑內的候選點（通常是路網節點）隨機挑出最多 count 位乘客（R1, R2, ...）。"""
    chosen = _shuffled_within(points, lambda point: point, center_x, center_y, radius_m, rng)
    return [Rider(id=f"R{i + 1}", x=x, y=y) for i, (x, y) in enumerate(chosen[:count])]


if __name__ == "__main__":
    from map_adapter import MapRoutingAdapter
    from matching_engine import MatchingEngine

    # 1. 抓取此刻全台北市營運中的公車位置
    live_buses = fetch_bus_positions()
    if not live_buses:
        raise SystemExit("目前沒有營運中的公車（可能是深夜收班時段）")
    newest_time = max(bus.data_time for bus in live_buses)
    print(f"=== 營運中公車 {len(live_buses)} 輛，資料時間 {newest_time:%Y-%m-%d %H:%M:%S} ===")

    # 2. 在大同區中心 1.5 公里內挑 6 輛公車當司機、4 個路口當乘客
    adapter = MapRoutingAdapter()
    demo_rng = random.Random(42)
    demo_x, demo_y, demo_radius = 121.5130, 25.0625, 1_500
    drivers, labels = pick_live_drivers(
        live_buses, demo_x, demo_y, demo_radius, 6, demo_rng,
        is_on_network=lambda x, y: adapter.snap_distance_m(x, y) <= 250,
    )
    nodes = [(data["x"], data["y"]) for _, data in adapter.graph.nodes(data=True)]
    riders = pick_random_riders(nodes, demo_x, demo_y, demo_radius, 4, demo_rng)

    print("\n=== Drivers（公車即時位置）===")
    for d in drivers:
        print(f"  {d.id}（公車 {labels[d.id]}）: ({d.x:.4f}, {d.y:.4f})")
    print("=== Riders（隨機路口）===")
    for r in riders:
        print(f"  {r.id}: ({r.x:.4f}, {r.y:.4f})")

    # 3. 真實路網成本 + 匈牙利演算法
    engine = MatchingEngine(cost_fn=adapter.build_batch_cost_fn(drivers))
    results = engine.match(drivers, riders)
    print("\n=== Match Results ===")
    for result in results:
        print(f"  {result.driver_id}（公車 {labels[result.driver_id]}）-> "
              f"{result.rider_id}  (行車距離 = {result.cost:.0f} 公尺)")
    print(f"\nTotal system cost: {sum(r.cost for r in results):.0f} 公尺")
