"""live_data 的單元測試（全部離線執行，不連網）。

涵蓋範圍：
- 公車資料解析：營運狀態過濾、無效座標與時間、過時紀錄、UTF-8 BOM
- 從 file:// 網址下載（走完整的下載 + 解壓流程，但不需要網路）
- 球面距離與依地圖縮放計算的取樣半徑
- 半徑內取樣司機與乘客：數量上限、路網外過濾、id 編號、可重現性
"""

from __future__ import annotations

import gzip
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from live_data import (
    BusPosition,
    fetch_bus_positions,
    haversine_m,
    parse_bus_data,
    pick_live_drivers,
    pick_random_riders,
    view_radius_m,
)

# 大同區中心（x=經度 lng, y=緯度 lat）
CENTER_X, CENTER_Y = 121.5130, 25.0625

# ---------------------------------------------------------------------------
# 測試輔助工具
# ---------------------------------------------------------------------------


def bus_record(
    bus_id: str,
    x: float | str = CENTER_X,
    y: float | str = CENTER_Y,
    *,
    duty: str = "1",
    status: str = "0",
    time: str = "2026-09-14 17:40:00",
) -> Dict[str, Any]:
    """產生一筆與 GetBusData 格式相同的公車紀錄（官方資料的數值都是字串）。"""
    return {
        "BusID": bus_id,
        "Longitude": str(x),
        "Latitude": str(y),
        "DutyStatus": duty,
        "BusStatus": status,
        "DataTime": time,
    }


def encode_feed(records: List[Dict[str, Any]]) -> bytes:
    """包成 gzip 壓縮、帶 UTF-8 BOM 的 JSON，與官方檔案格式一致。"""
    payload = json.dumps({"EssentialInfo": {}, "BusInfo": records}, ensure_ascii=False)
    return gzip.compress(payload.encode("utf-8-sig"))


def bus_ids(buses: List[BusPosition]) -> Set[str]:
    return {bus.bus_id for bus in buses}


def make_bus(bus_id: str, x: float, y: float) -> BusPosition:
    return BusPosition(bus_id=bus_id, x=x, y=y, data_time=datetime(2026, 9, 14, 17, 40))


def always_on_network(x: float, y: float) -> bool:
    return True


# 中心附近 5 輛（每輛往東北多偏移約 150 公尺，最遠約 750 公尺）與約 15 公里外 1 輛
NEAR_BUSES = [
    make_bus(f"N{i}", CENTER_X + i * 0.001, CENTER_Y + i * 0.001) for i in range(1, 6)
]
FAR_BUS = make_bus("FAR", CENTER_X + 0.1, CENTER_Y + 0.1)
ALL_BUSES = NEAR_BUSES + [FAR_BUS]


# ---------------------------------------------------------------------------
# 公車資料解析
# ---------------------------------------------------------------------------


class TestParseBusData:
    def test_keeps_only_operating_normal_buses(self) -> None:
        """只留 DutyStatus=1（勤務中）且 BusStatus=0（行車正常）的公車。"""
        raw = encode_feed(
            [
                bus_record("OK"),
                bus_record("NOT_DEPARTED", duty="0"),
                bus_record("FINISHED", duty="2"),
                bus_record("OFF_SERVICE", status="99"),
                bus_record("BREAKDOWN", status="2"),
            ]
        )
        assert bus_ids(parse_bus_data(raw)) == {"OK"}

    def test_coordinates_follow_lng_lat_convention(self) -> None:
        [bus] = parse_bus_data(encode_feed([bus_record("A", x=121.52, y=25.03)]))
        assert bus.x == pytest.approx(121.52)  # x = 經度
        assert bus.y == pytest.approx(25.03)  # y = 緯度

    def test_skips_invalid_coordinates(self) -> None:
        raw = encode_feed(
            [
                bus_record("OK"),
                bus_record("EMPTY", x=""),
                bus_record("ZERO", x=0, y=0),
                bus_record("TEXT", y="abc"),
            ]
        )
        assert bus_ids(parse_bus_data(raw)) == {"OK"}

    def test_skips_unparseable_time(self) -> None:
        raw = encode_feed([bus_record("OK"), bus_record("NO_TIME", time="")])
        assert bus_ids(parse_bus_data(raw)) == {"OK"}

    def test_drops_stale_records_relative_to_newest(self) -> None:
        """以資料中最新一筆為基準，超過 max_age_s 沒回報位置的公車視為過時。"""
        raw = encode_feed(
            [
                bus_record("NEWEST", time="2026-09-14 17:40:00"),
                bus_record("2_MIN_AGO", time="2026-09-14 17:38:00"),
                bus_record("4_MIN_AGO", time="2026-09-14 17:36:00"),
            ]
        )
        assert bus_ids(parse_bus_data(raw, max_age_s=180)) == {"NEWEST", "2_MIN_AGO"}

    def test_empty_feed(self) -> None:
        assert parse_bus_data(encode_feed([])) == []

    def test_invalid_or_truncated_gzip_raises_value_error(self) -> None:
        """下載到非 gzip 或被截斷的檔案時統一拋 ValueError，呼叫端只需處理一種格式錯誤。"""
        with pytest.raises(ValueError):
            parse_bus_data(b"<html>not gzip</html>")
        with pytest.raises(ValueError):
            parse_bus_data(encode_feed([bus_record("A")])[:-20])


class TestFetchBusPositions:
    def test_downloads_and_parses_feed(self, tmp_path: Path) -> None:
        """用 file:// 網址走完整的下載流程，不需要網路。"""
        feed = tmp_path / "GetBusData.gz"
        feed.write_bytes(encode_feed([bus_record("A"), bus_record("B", status="99")]))
        assert bus_ids(fetch_bus_positions(url=feed.as_uri())) == {"A"}


# ---------------------------------------------------------------------------
# 距離與取樣半徑
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_same_point_is_zero(self) -> None:
        assert haversine_m(CENTER_X, CENTER_Y, CENTER_X, CENTER_Y) == pytest.approx(0.0)

    def test_one_degree_of_latitude(self) -> None:
        """緯度差 1 度約 111.2 公里。"""
        assert haversine_m(121.5, 25.0, 121.5, 26.0) == pytest.approx(111_195, rel=1e-3)

    def test_arguments_are_lng_lat(self) -> None:
        """經度差 1 度在北緯 25 度約 100.8 公里（赤道的 cos 25° 倍）；
        若誤把參數當成 (lat, lng)，結果會差很多。"""
        assert haversine_m(121.0, 25.0, 122.0, 25.0) == pytest.approx(100_780, rel=1e-2)


class TestViewRadius:
    def test_fits_inside_visible_map_at_zoom_14(self) -> None:
        """zoom 14 的台北，550px 高的地圖半個畫面約 2,380 公尺，半徑要比它小。"""
        radius = view_radius_m(CENTER_Y, zoom=14, map_height_px=550)
        assert 1_000 < radius < 2_380

    def test_zooming_in_one_level_halves_radius(self) -> None:
        r15 = view_radius_m(CENTER_Y, zoom=15, map_height_px=550)
        r16 = view_radius_m(CENTER_Y, zoom=16, map_height_px=550)
        assert r16 == pytest.approx(r15 / 2)

    def test_clamped_to_max_when_zoomed_out(self) -> None:
        assert view_radius_m(CENTER_Y, zoom=10, map_height_px=550, max_m=2_000) == 2_000

    def test_clamped_to_min_when_zoomed_in(self) -> None:
        assert view_radius_m(CENTER_Y, zoom=19, map_height_px=550, min_m=300) == 300


# ---------------------------------------------------------------------------
# 取樣司機（公車）
# ---------------------------------------------------------------------------


class TestPickLiveDrivers:
    def test_only_buses_within_radius(self) -> None:
        drivers, labels = pick_live_drivers(
            ALL_BUSES, CENTER_X, CENTER_Y, 1_000, 10, random.Random(0), always_on_network
        )
        assert len(drivers) == 5
        assert set(labels.values()) == {"N1", "N2", "N3", "N4", "N5"}

    def test_respects_count(self) -> None:
        drivers, _ = pick_live_drivers(
            ALL_BUSES, CENTER_X, CENTER_Y, 1_000, 3, random.Random(0), always_on_network
        )
        assert len(drivers) == 3

    def test_ids_sequential_and_labels_match_coordinates(self) -> None:
        """司機 id 依序為 D1..Dn，labels 對應回原本的公車，座標沿用 x=lng, y=lat。"""
        drivers, labels = pick_live_drivers(
            ALL_BUSES, CENTER_X, CENTER_Y, 1_000, 3, random.Random(0), always_on_network
        )
        assert [d.id for d in drivers] == ["D1", "D2", "D3"]
        buses_by_id = {bus.bus_id: bus for bus in ALL_BUSES}
        for driver in drivers:
            bus = buses_by_id[labels[driver.id]]
            assert (driver.x, driver.y) == (bus.x, bus.y)

    def test_skips_buses_off_network(self) -> None:
        """路網外的公車（例如在新北市）即使在半徑內也不能當司機。"""
        off_network = {(bus.x, bus.y) for bus in NEAR_BUSES[:2]}
        _, labels = pick_live_drivers(
            ALL_BUSES, CENTER_X, CENTER_Y, 1_000, 10, random.Random(0),
            is_on_network=lambda x, y: (x, y) not in off_network,
        )
        assert set(labels.values()) == {"N3", "N4", "N5"}

    @pytest.mark.parametrize("seed", range(20))
    def test_off_network_buses_do_not_use_up_count(self, seed: int) -> None:
        """被略過的公車不佔名額：路網上剛好 3 輛、count=3，每次都要取滿 3 輛。"""
        off_network = {(bus.x, bus.y) for bus in NEAR_BUSES[:2]}
        drivers, _ = pick_live_drivers(
            ALL_BUSES, CENTER_X, CENTER_Y, 1_000, 3, random.Random(seed),
            is_on_network=lambda x, y: (x, y) not in off_network,
        )
        assert len(drivers) == 3

    def test_same_seed_same_result(self) -> None:
        def pick(seed: int) -> Dict[str, str]:
            return pick_live_drivers(
                ALL_BUSES, CENTER_X, CENTER_Y, 1_000, 3, random.Random(seed), always_on_network
            )[1]

        assert pick(7) == pick(7)

    def test_no_buses_in_range(self) -> None:
        assert pick_live_drivers(
            [FAR_BUS], CENTER_X, CENTER_Y, 1_000, 5, random.Random(0), always_on_network
        ) == ([], {})


# ---------------------------------------------------------------------------
# 取樣乘客（路網節點）
# ---------------------------------------------------------------------------


class TestPickRandomRiders:
    # 中心附近 6 個點（最遠約 900 公尺）與 10 公里外 1 個點
    NEAR_POINTS = [(CENTER_X + i * 0.001, CENTER_Y - i * 0.001) for i in range(1, 7)]
    FAR_POINTS = [(CENTER_X - 0.1, CENTER_Y)]

    def test_within_radius_and_count(self) -> None:
        riders = pick_random_riders(
            self.NEAR_POINTS + self.FAR_POINTS, CENTER_X, CENTER_Y, 1_000, 4, random.Random(0)
        )
        assert len(riders) == 4
        assert all(haversine_m(CENTER_X, CENTER_Y, r.x, r.y) <= 1_000 for r in riders)

    def test_ids_sequential_and_no_duplicates(self) -> None:
        riders = pick_random_riders(self.NEAR_POINTS, CENTER_X, CENTER_Y, 1_000, 3, random.Random(0))
        assert [r.id for r in riders] == ["R1", "R2", "R3"]
        assert len({(r.x, r.y) for r in riders}) == 3

    def test_fewer_candidates_than_count(self) -> None:
        riders = pick_random_riders(
            self.NEAR_POINTS[:2] + self.FAR_POINTS, CENTER_X, CENTER_Y, 1_000, 4, random.Random(0)
        )
        assert len(riders) == 2
