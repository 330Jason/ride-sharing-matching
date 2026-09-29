"""MapRoutingAdapter 路徑繪製的單元測試（離線執行，不連網）。

用一張小型人工路網（存成 graphml 後由 MapRoutingAdapter 正常載入），驗證
get_routing_path 畫出來的折線：
- 起點是司機在路段上的實際位置，終點是乘客可上車的路邊位置（都不是路口）
- 中間全程沿著道路的實際形狀走，不橫切街廓
- 單行道只能順向行駛

人工路網（x=經度 lng, y=緯度 lat）：

    E(121.510,25.060) ←── D(121.520,25.060)
        │                      ↑
        ↓                      │
    A ══════ B ────→ C ────────┘
  (121.500) (121.510)  (121.520)  ← 三點都在 lat 25.050

- A↔B 雙向，實際道路往北繞一個ㄇ字（直線連 A、B 會切過街廓）
- B→C 單行道，沒有 geometry（單純直線）
- C→D→E→B 是回到 B 的單行迴路
- P→Q 是另一個互不相連的區塊，用來測不可達
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import LineString

from map_adapter import MapRoutingAdapter
from matching_engine import Driver, Rider

A_XY = (121.500, 25.050)
B_XY = (121.510, 25.050)
C_XY = (121.520, 25.050)
D_XY = (121.520, 25.060)
E_XY = (121.510, 25.060)
P_XY = (121.560, 25.080)
Q_XY = (121.570, 25.080)

# A→B 的實際道路：往北繞一個ㄇ字
AB_SHAPE = [A_XY, (121.500, 25.056), (121.510, 25.056), B_XY]
BA_SHAPE = AB_SHAPE[::-1]
# B→C 上的平行繞路（只有在 parallel_edge=True 時才加入）
BC_DETOUR = [B_XY, (121.515, 25.040), C_XY]


def to_latlng(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(lng, lat) 串列轉成 folium 用的 (lat, lng) 串列。"""
    return [(y, x) for x, y in points]


def build_graph(
    tmp_path: Path, *, reverse_geometry: bool = False, parallel_edge: bool = False
) -> str:
    """產生小型人工路網並存成 graphml，回傳檔案路徑。"""
    graph = nx.MultiDiGraph(crs="epsg:4326", simplified=True)
    for node, (x, y) in [
        (1, A_XY), (2, B_XY), (3, C_XY), (4, D_XY), (5, E_XY), (6, P_XY), (7, Q_XY)
    ]:
        graph.add_node(node, x=x, y=y)

    graph.add_edge(1, 2, key=0, length=3_000.0, geometry=LineString(AB_SHAPE))
    # 反向邊：geometry 正常以行進方向儲存，reverse_geometry 時故意存成相反方向
    graph.add_edge(
        2, 1, key=0, length=3_000.0,
        geometry=LineString(AB_SHAPE if reverse_geometry else BA_SHAPE),
    )
    graph.add_edge(2, 3, key=0, length=1_000.0)  # 單行、無 geometry
    graph.add_edge(3, 4, key=0, length=1_100.0)  # 迴路：C→D→E→B
    graph.add_edge(4, 5, key=0, length=1_100.0)
    graph.add_edge(5, 2, key=0, length=1_100.0)
    graph.add_edge(6, 7, key=0, length=500.0)    # 另一個互不相連的區塊
    if parallel_edge:
        graph.add_edge(2, 3, key=1, length=9_000.0, geometry=LineString(BC_DETOUR))

    path = tmp_path / "tiny.graphml"
    ox.save_graphml(graph, path)
    return str(path)


def driver_at(xy: Tuple[float, float]) -> Driver:
    return Driver(id="D1", x=xy[0], y=xy[1])


def rider_at(xy: Tuple[float, float]) -> Rider:
    return Rider(id="R1", x=xy[0], y=xy[1])


@pytest.fixture
def adapter(tmp_path: Path) -> MapRoutingAdapter:
    return MapRoutingAdapter(cache_path=build_graph(tmp_path))


class TestPathEndpoints:
    def test_starts_at_driver_position_not_junction(self, adapter: MapRoutingAdapter) -> None:
        """司機停在路段中間時，路線要從他的實際位置起算，而不是從最近路口。"""
        driver_point = (121.505, 25.056)  # ㄇ字頂邊的中間
        path = adapter.get_routing_path(driver_at(driver_point), rider_at(C_XY))
        assert path[0] == to_latlng([driver_point])[0]
        assert path[-1] == to_latlng([C_XY])[0]
        assert path == to_latlng([driver_point, (121.510, 25.056), B_XY, C_XY])

    def test_ends_at_roadside_of_rider_in_a_building(self, adapter: MapRoutingAdapter) -> None:
        """乘客在民宅（不在路上）時，路線終點是該處最近的路邊上車點。"""
        rider_point = (121.505, 25.0555)   # 頂邊往南約 55 公尺的街廓內
        curb_point = (121.505, 25.056)     # 對應的路邊位置
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(rider_point))
        assert path[-1] == to_latlng([curb_point])[0]
        assert to_latlng([rider_point])[0] not in path  # 不會畫進民宅裡
        assert path == to_latlng([A_XY, (121.500, 25.056), curb_point])

    def test_both_ends_on_the_same_segment(self, adapter: MapRoutingAdapter) -> None:
        """司機與乘客在同一條路段上且順向：直接沿該路段連起來，不繞路。"""
        path = adapter.get_routing_path(
            driver_at((121.512, 25.050)), rider_at((121.518, 25.050))
        )
        assert path == to_latlng([(121.512, 25.050), (121.518, 25.050)])


class TestPathFollowsStreets:
    def test_follows_road_shape_between_junctions(self, adapter: MapRoutingAdapter) -> None:
        """路口之間照道路實際形狀走，不把兩個路口連成直線。"""
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(B_XY))
        assert path == to_latlng(AB_SHAPE)

    def test_segment_without_geometry_is_a_straight_line(self, adapter: MapRoutingAdapter) -> None:
        path = adapter.get_routing_path(driver_at(B_XY), rider_at(C_XY))
        assert path == to_latlng([B_XY, C_XY])

    def test_junction_not_duplicated_between_segments(self, adapter: MapRoutingAdapter) -> None:
        """跨兩段路時，中間的路口只出現一次。"""
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(C_XY))
        assert path == to_latlng(AB_SHAPE + [C_XY])

    def test_reversed_geometry_is_flipped(self, tmp_path: Path) -> None:
        """geometry 以相反方向儲存時，畫出來仍須從起點走向終點。"""
        adapter = MapRoutingAdapter(cache_path=build_graph(tmp_path, reverse_geometry=True))
        path = adapter.get_routing_path(driver_at(B_XY), rider_at(A_XY))
        assert path == to_latlng(BA_SHAPE)

    def test_uses_same_edge_as_shortest_path(self, tmp_path: Path) -> None:
        """B→C 有兩條平行路段時，畫線要跟最短路徑選同一條（較短的那條）。"""
        adapter = MapRoutingAdapter(cache_path=build_graph(tmp_path, parallel_edge=True))
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(C_XY))
        assert path == to_latlng(AB_SHAPE + [C_XY])
        assert to_latlng(BC_DETOUR)[1] not in path


class TestOneWayAndUnreachable:
    def test_never_travels_against_a_one_way(self, adapter: MapRoutingAdapter) -> None:
        """司機在單行道 B→C 上，要回到 B 只能順向走完再繞迴路 C→D→E→B。"""
        driver_point = (121.515, 25.050)
        path = adapter.get_routing_path(driver_at(driver_point), rider_at(B_XY))
        assert path == to_latlng([driver_point, C_XY, D_XY, E_XY, B_XY])

    def test_unreachable_returns_empty(self, adapter: MapRoutingAdapter) -> None:
        """乘客落在互不相連的區塊：沒有路可走，回傳空串列。"""
        assert adapter.get_routing_path(driver_at(A_XY), rider_at((121.565, 25.080))) == []
