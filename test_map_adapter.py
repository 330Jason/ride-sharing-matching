"""MapRoutingAdapter 路徑繪製的單元測試（離線執行，不連網）。

用一張小型人工路網（存成 graphml 後由 MapRoutingAdapter 正常載入），
驗證 get_routing_path 畫出來的折線沿著道路的實際形狀走，而不是把路口
直接連成直線——後者在彎曲或繞行的路段上會橫穿街廓。
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

# 人工路網的節點（x=經度 lng, y=緯度 lat）
A, B, C, D = 1, 2, 3, 4
A_XY = (121.500, 25.050)
B_XY = (121.510, 25.050)
C_XY = (121.520, 25.050)
D_XY = (121.530, 25.060)  # 孤立節點，沒有任何路段相連
# A→B 這段路實際上往北繞一個彎；直線連 A、B 會切過中間的街廓
AB_SHAPE = [A_XY, (121.503, 25.056), (121.507, 25.056), B_XY]
# 平行路段用的另一條更長的繞路形狀
AB_DETOUR = [A_XY, (121.505, 25.040), B_XY]


def to_latlng(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """(lng, lat) 串列轉成 folium 用的 (lat, lng) 串列。"""
    return [(y, x) for x, y in points]


def build_graph(
    tmp_path: Path, *, reverse_geometry: bool = False, parallel_edge: bool = False
) -> str:
    """產生小型人工路網並存成 graphml，回傳檔案路徑。"""
    graph = nx.MultiDiGraph(crs="epsg:4326", simplified=True)
    for node, (x, y) in [(A, A_XY), (B, B_XY), (C, C_XY), (D, D_XY)]:
        graph.add_node(node, x=x, y=y)

    shape = AB_SHAPE[::-1] if reverse_geometry else AB_SHAPE
    graph.add_edge(A, B, key=0, length=1_500.0, geometry=LineString(shape))
    graph.add_edge(B, C, key=0, length=1_000.0)  # 沒有 geometry：單純的直線路段
    if parallel_edge:
        # 更長的繞路：shortest_path 會選上面那條，畫線也必須跟著選同一條
        graph.add_edge(A, B, key=1, length=9_000.0, geometry=LineString(AB_DETOUR))

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


class TestRoutingPathFollowsStreets:
    def test_includes_road_shape_points(self, adapter: MapRoutingAdapter) -> None:
        """有 geometry 的路段：折線要照道路實際形狀，含中間的轉折點。"""
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(B_XY))
        assert path == to_latlng(AB_SHAPE)

    def test_falls_back_to_nodes_without_geometry(self, adapter: MapRoutingAdapter) -> None:
        """沒有 geometry 的路段：仍以兩端節點連線。"""
        path = adapter.get_routing_path(driver_at(B_XY), rider_at(C_XY))
        assert path == to_latlng([B_XY, C_XY])

    def test_junction_not_duplicated_between_segments(self, adapter: MapRoutingAdapter) -> None:
        """跨兩段路時，中間的路口只出現一次。"""
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(C_XY))
        assert path == to_latlng(AB_SHAPE + [C_XY])

    def test_unreachable_returns_empty(self, adapter: MapRoutingAdapter) -> None:
        assert adapter.get_routing_path(driver_at(A_XY), rider_at(D_XY)) == []

    def test_reversed_geometry_is_flipped(self, tmp_path: Path) -> None:
        """geometry 以相反方向儲存時，畫出來仍須從起點走向終點。"""
        adapter = MapRoutingAdapter(cache_path=build_graph(tmp_path, reverse_geometry=True))
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(B_XY))
        assert path[0] == to_latlng([A_XY])[0]
        assert path[-1] == to_latlng([B_XY])[0]
        assert path == to_latlng(AB_SHAPE)

    def test_uses_same_edge_as_shortest_path(self, tmp_path: Path) -> None:
        """兩條平行路段之間，畫線要跟最短路徑選同一條（較短的那條）。"""
        adapter = MapRoutingAdapter(cache_path=build_graph(tmp_path, parallel_edge=True))
        path = adapter.get_routing_path(driver_at(A_XY), rider_at(B_XY))
        assert path == to_latlng(AB_SHAPE)
        assert to_latlng(AB_DETOUR)[1] not in path
