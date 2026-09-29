"""即時共乘派單與最佳化配對系統 — 第二階段：真實路網轉接器。

MapRoutingAdapter 把 OpenStreetMap 的真實街道路網包裝成
MatchingEngine 可注入的成本函式 (cost_fn)：
- 座標慣例：Driver/Rider 的 x 為經度 (lng)、y 為緯度 (lat)。
- 成本單位：真實街道最短行車距離（公尺）。
- 路網不通（NetworkXNoPath）時回傳 float('inf')，
  由 MatchingEngine 既有的不可達處理機制跳過該組合。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
from shapely.ops import substring

from matching_engine import Driver, Rider

# 投影點與路段端點相距小於這個距離（經緯度）就視為就在該路口上。
# 約等於 0.1 公釐，只用來吸收浮點誤差。
ENDPOINT_TOLERANCE = 1e-9


class EdgeSnap(NamedTuple):
    """一個座標投影到路網上的結果。

    u、v、key 是該路段在圖上的識別；line 是以 u→v 方向排列的道路形狀；
    along 是投影點沿著 line 前進的距離（經緯度單位，僅用於切線段與算比例）。
    """

    u: int
    v: int
    key: int
    line: LineString
    along: float


class MapRoutingAdapter:
    """真實地圖路由轉接器：載入 OSM 汽車路網並計算最短行車距離。"""

    def __init__(
        self,
        place_name: str = "Taipei, Taiwan",
        cache_path: str = "taipei_city.graphml",
    ) -> None:
        """載入路網圖資。

        本地若已有 cache_path 的 .graphml 快取就直接讀取；
        否則才向 OSM 下載（耗時），並存檔供之後重複使用。
        """
        cache = Path(cache_path)
        if cache.exists():
            print(f"[MapRoutingAdapter] 從快取載入路網: {cache}")
            self.graph: nx.MultiDiGraph = ox.load_graphml(cache)
        else:
            print(f"[MapRoutingAdapter] 下載路網中: {place_name} ...")
            self.graph = ox.graph_from_place(place_name, network_type="drive")
            ox.save_graphml(self.graph, cache)
            print(f"[MapRoutingAdapter] 已存入快取: {cache}")

        # 同一個座標在一輪派單中會被查詢多次（n×m 個組合），
        # 快取「座標 → 最近節點」避免重複的最近鄰搜尋。
        self._node_cache: Dict[Tuple[float, float], int] = {}
        # 「座標 → 最近路段投影」也快取：nearest_edges 每次呼叫都要重建
        # 空間索引，單點就要 0.15 秒左右。封路改變時會一併清掉。
        self._edge_cache: Dict[Tuple[float, float], Optional[EdgeSnap]] = {}

        # 套用封路後的「工作圖」：所有路由計算都走這張圖。
        # 尚未封路時直接指向原圖（共用、不複製），apply_roadblocks
        # 會在需要時換成移除了封鎖節點的副本——self.graph 永不被更動。
        self.working_graph: nx.MultiDiGraph = self.graph

    def apply_roadblocks(
        self, roadblock_coords: List[Tuple[float, float]]
    ) -> None:
        """套用動態封路：在原圖的乾淨副本上移除被封鎖路口對應的節點。

        roadblock_coords 為 [(lat, lng), ...]（與前端 folium 點擊順序一致）。
        刻意絕不更動 self.graph——每次都從原圖重新複製一份 working_graph，
        所以解除封路（傳入空清單）可完全還原，重複封路也不會累積污染。
        移除節點會連帶切斷其所有相連的邊，後續路由自然繞道。
        """
        # 投影結果綁在 working_graph 上，換圖就必須重算
        self._edge_cache.clear()

        if not roadblock_coords:
            # 無封路：直接共用原圖，省下整張圖的複製成本
            self.working_graph = self.graph
            return

        working_graph = self.graph.copy()
        # _nearest_node 吃 (lng, lat)，封路座標是 (lat, lng) 需對調
        blocked_nodes = {
            self._nearest_node(lng, lat) for lat, lng in roadblock_coords
        }
        # remove_nodes_from 會自動忽略不存在的節點（重複封同一路口也安全）
        working_graph.remove_nodes_from(blocked_nodes)
        self.working_graph = working_graph

    def _nearest_node(self, x: float, y: float) -> int:
        """把 (lng, lat) 座標映射到路網上最近的節點 id。"""
        key = (x, y)
        if key not in self._node_cache:
            self._node_cache[key] = ox.distance.nearest_nodes(self.graph, X=x, Y=y)
        return self._node_cache[key]

    def snap_distance_m(self, x: float, y: float) -> float:
        """(lng, lat) 座標到最近路網節點的直線距離（公尺）。

        nearest_nodes 對「路網範圍外」的座標（河面、其他行政區）
        仍會硬吸附到最近的邊界節點，導致算出來的路徑遠離原始座標。
        呼叫端可用這個距離判斷座標是否真的落在路網涵蓋範圍內。
        """
        _, dist = ox.distance.nearest_nodes(
            self.graph, X=x, Y=y, return_dist=True
        )
        return float(dist)

    def shortest_distance(
        self, orig_x: float, orig_y: float, dest_x: float, dest_y: float
    ) -> float:
        """任意兩個 (lng, lat) 座標間的最短行車距離（公尺）。

        路網不通時回傳 inf（代表不可達）。
        """
        source = self._nearest_node(orig_x, orig_y)
        target = self._nearest_node(dest_x, dest_y)
        try:
            return float(
                nx.shortest_path_length(
                    self.working_graph, source, target, weight="length"
                )
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # NodeNotFound：起訖點剛好落在被封鎖（已移除）的節點上
            return float("inf")

    def get_routing_cost(self, driver: Driver, rider: Rider) -> float:
        """計算司機到乘客「上車點」的最短行車距離（公尺）。

        簽名與 MatchingEngine 的 cost_fn 相容，可直接注入：
        MatchingEngine(cost_fn=adapter.get_routing_cost)。
        """
        return self.shortest_distance(driver.x, driver.y, rider.x, rider.y)

    def get_routing_path(
        self, driver: Driver, rider: Rider
    ) -> List[Tuple[float, float]]:
        """司機到乘客上車點的真實街道路徑，回傳沿途座標串列。

        座標格式為 [(lat, lng), ...]——刻意採用 folium/leaflet 的
        (緯度, 經度) 順序，前端可直接餵給 folium.PolyLine 繪製。
        """
        return self.route_shape(driver.x, driver.y, rider.x, rider.y)

    def route_shape(
        self, orig_x: float, orig_y: float, dest_x: float, dest_y: float
    ) -> List[Tuple[float, float]]:
        """兩個 (lng, lat) 座標之間的行車路線形狀，回傳 [(lat, lng), ...]。

        路線從「起點所在路段上的實際位置」出發，全程沿著道路的實際形狀，
        走到「終點最近的路邊位置」為止：
        - 起訖點都投影到最近的路段（不是最近的路口）。乘客若在民宅裡，
          終點就是該民宅的路邊上車點。
        - 從投影點沿著該路段走到路口才接上主路徑；單行道只走順向，
          要回頭就得繞路。
        - 起訖在同一條路段且順向時，直接沿該路段連起來。
        走 working_graph，因此會自動繞過已封鎖的路口；不可達時回傳空串列。
        """
        origin = self._snap_to_edge(orig_x, orig_y)
        destination = self._snap_to_edge(dest_x, dest_y)
        if origin is None or destination is None:
            return []

        same_edge = (origin.u, origin.v, origin.key) == (
            destination.u, destination.v, destination.key
        )
        if same_edge and origin.along <= destination.along:
            return self._to_latlng(
                self._coords(substring(origin.line, origin.along, destination.along))
            )

        best_cost = float("inf")
        best_shape: List[Tuple[float, float]] = []
        for start_node, head_shape, head_cost in self._departures(origin):
            for end_node, tail_shape, tail_cost in self._arrivals(destination):
                try:
                    middle_cost, node_ids = nx.single_source_dijkstra(
                        self.working_graph, start_node, end_node, weight="length"
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                total = head_cost + float(middle_cost) + tail_cost
                if total >= best_cost:
                    continue
                middle_shape: List[Tuple[float, float]] = []
                for u, v in zip(node_ids, node_ids[1:]):
                    middle_shape.extend(self._edge_shape(u, v))
                best_cost = total
                best_shape = head_shape + middle_shape + tail_shape

        return self._to_latlng(best_shape)

    # ------------------------------------------------------------------
    # 路段投影與形狀
    # ------------------------------------------------------------------

    def _snap_to_edge(self, x: float, y: float) -> Optional[EdgeSnap]:
        """把 (lng, lat) 投影到最近的「路段」上（不是最近的路口）。"""
        key = (x, y)
        if key not in self._edge_cache:
            try:
                u, v, edge_key = ox.distance.nearest_edges(
                    self.working_graph, X=x, Y=y
                )
                line = self._edge_line(u, v, edge_key)
                along = float(line.project(_point(x, y)))
                self._edge_cache[key] = EdgeSnap(u, v, edge_key, line, along)
            except (ValueError, KeyError, nx.NetworkXError):
                self._edge_cache[key] = None
        return self._edge_cache[key]

    def _edge_line(self, u: int, v: int, key: int) -> LineString:
        """u→v 這條路段的形狀，方向一律由 u 指向 v。

        沒有 geometry 的路段（osmnx 未簡化的直線段）退回兩端路口的直線。
        """
        data = self.working_graph.edges[u, v, key]
        start = (
            self.working_graph.nodes[u]["x"], self.working_graph.nodes[u]["y"]
        )
        end = (
            self.working_graph.nodes[v]["x"], self.working_graph.nodes[v]["y"]
        )

        geometry = data.get("geometry")
        if geometry is None:
            return LineString([start, end])

        coords = [(float(px), float(py)) for px, py in geometry.coords]

        def gap_to_start(point: Tuple[float, float]) -> float:
            return (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2

        # 雙向道路的兩個方向可能共用同一條 geometry，存的方向不一定相同；
        # 頭端離 u 較遠就整條反轉，確保形狀是從 u 走向 v
        if gap_to_start(coords[0]) > gap_to_start(coords[-1]):
            coords.reverse()
        return LineString(coords)

    def _edge_shape(self, u: int, v: int) -> List[Tuple[float, float]]:
        """u→v 的 (lng, lat) 形狀點串列；平行路段取與最短路徑相同的最短那條。"""
        edges = self.working_graph.get_edge_data(u, v)
        key = min(edges, key=lambda k: edges[k].get("length", float("inf")))
        return list(self._edge_line(u, v, key).coords)

    def _edge_length_m(self, snap: EdgeSnap) -> float:
        """投影所在路段的實際長度（公尺）。"""
        data = self.working_graph.edges[snap.u, snap.v, snap.key]
        return float(data.get("length", 0.0))

    def _departures(
        self, snap: EdgeSnap
    ) -> List[Tuple[int, List[Tuple[float, float]], float]]:
        """從投影點出發可以先到哪些路口。

        回傳 [(路口 id, 到該路口的形狀, 距離公尺), ...]。
        順向一定可走；逆向只有在該路段存在反向邊（雙向道路）時才可走，
        因此單行道不會被逆向行駛。
        """
        total = snap.line.length
        first, last = self._coords(snap.line)[0], self._coords(snap.line)[-1]
        if total <= 0 or snap.along <= ENDPOINT_TOLERANCE:
            return [(snap.u, [first], 0.0)]
        if snap.along >= total - ENDPOINT_TOLERANCE:
            return [(snap.v, [last], 0.0)]

        length_m = self._edge_length_m(snap)
        ratio = snap.along / total
        options = [(
            snap.v,
            self._coords(substring(snap.line, snap.along, total)),
            length_m * (1 - ratio),
        )]
        if self.working_graph.has_edge(snap.v, snap.u):
            backward = self._coords(substring(snap.line, 0, snap.along))[::-1]
            options.append((snap.u, backward, length_m * ratio))
        return options

    def _arrivals(
        self, snap: EdgeSnap
    ) -> List[Tuple[int, List[Tuple[float, float]], float]]:
        """可以從哪些路口沿路段走到投影點（上車點）。

        回傳 [(路口 id, 從該路口到投影點的形狀, 距離公尺), ...]。
        """
        total = snap.line.length
        first, last = self._coords(snap.line)[0], self._coords(snap.line)[-1]
        if total <= 0 or snap.along <= ENDPOINT_TOLERANCE:
            return [(snap.u, [first], 0.0)]
        if snap.along >= total - ENDPOINT_TOLERANCE:
            return [(snap.v, [last], 0.0)]

        length_m = self._edge_length_m(snap)
        ratio = snap.along / total
        options = [(
            snap.u,
            self._coords(substring(snap.line, 0, snap.along)),
            length_m * ratio,
        )]
        if self.working_graph.has_edge(snap.v, snap.u):
            backward = self._coords(substring(snap.line, snap.along, total))[::-1]
            options.append((snap.v, backward, length_m * (1 - ratio)))
        return options

    @staticmethod
    def _coords(geometry) -> List[Tuple[float, float]]:
        """shapely 幾何轉成 (lng, lat) 串列；起訖相同時 substring 會回傳一個點。"""
        return [(float(x), float(y)) for x, y in geometry.coords]

    @staticmethod
    def _to_latlng(
        points: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """(lng, lat) 串列轉成 folium 的 (lat, lng)，並去掉連續重複的點。"""
        path: List[Tuple[float, float]] = []
        for x, y in points:
            point = (y, x)
            if not path or path[-1] != point:
                path.append(point)
        return path

    def build_batch_cost_fn(
        self, drivers: List[Driver]
    ) -> Callable[[Driver, Rider], float]:
        """為一個派單批次建立高效版成本函式（get_routing_cost 的批次替代品）。

        get_routing_cost 會對 n×m 個司機—乘客組合各跑一次最短路徑搜尋；
        這裡改為每個「司機所在節點」只跑一次
        nx.single_source_dijkstra_path_length，一次算出該節點到全路網
        所有節點的距離，之後 cost_fn 只剩 O(1) 查表，
        整體複雜度從 O(n×m × Dijkstra) 降為 O(n × Dijkstra)。

        注意：回傳的函式只對「建立批次時傳入的 drivers」有效。
        距離表中查不到目標節點代表路網不通，回傳 inf。
        """
        distances_by_source: Dict[int, Dict[int, float]] = {}
        for driver in drivers:
            source = self._nearest_node(driver.x, driver.y)
            if source not in distances_by_source:
                try:
                    distances_by_source[source] = (
                        nx.single_source_dijkstra_path_length(
                            self.working_graph, source, weight="length"
                        )
                    )
                except nx.NodeNotFound:
                    # 司機所在路口本身被封鎖：到任何乘客都不可達
                    distances_by_source[source] = {}

        def batch_cost_fn(driver: Driver, rider: Rider) -> float:
            source = self._nearest_node(driver.x, driver.y)
            target = self._nearest_node(rider.x, rider.y)
            return float(distances_by_source[source].get(target, float("inf")))

        return batch_cost_fn


def _point(x: float, y: float):
    """延後 import，避免模組層多一個 shapely 名稱。"""
    from shapely.geometry import Point

    return Point(x, y)


if __name__ == "__main__":
    from matching_engine import MatchingEngine

    # 1. 載入台北市路網（第一次執行會下載，之後走 .graphml 快取）
    adapter = MapRoutingAdapter()
    print(
        f"[MapRoutingAdapter] 路網節點數: {len(adapter.graph.nodes)}, "
        f"路段數: {len(adapter.graph.edges)}"
    )

    # 2. 真實地標座標——大同區一帶（x=經度 lng, y=緯度 lat）
    drivers = [
        Driver(id="D1-台北車站", x=121.5155, y=25.0495),   # 台北車站北側
        Driver(id="D2-大橋頭站", x=121.5128, y=25.0630),   # 大橋頭捷運站
    ]
    riders = [
        Rider(id="R1-寧夏夜市", x=121.5155, y=25.0565),    # 寧夏夜市
        Rider(id="R2-迪化街", x=121.5100, y=25.0555),      # 迪化街商圈
    ]

    # 3. 把真實路網成本注入第一階段的 MatchingEngine
    engine = MatchingEngine(cost_fn=adapter.get_routing_cost)

    cost_matrix = engine.build_cost_matrix(drivers, riders)
    print("\n=== Cost Matrix（公尺, rows=Drivers, cols=Riders）===")
    header = "              " + "".join(f"{r.id:>14}" for r in riders)
    print(header)
    for d, row in zip(drivers, cost_matrix):
        print(f"  {d.id:<12}" + "".join(f"{cost:>14.0f}" for cost in row))

    # 4. 執行配對並印出結果
    results = engine.match(drivers, riders)

    drivers_by_id = {d.id: d for d in drivers}
    riders_by_id = {r.id: r for r in riders}

    print("\n=== Match Results ===")
    for result in results:
        path = adapter.get_routing_path(
            drivers_by_id[result.driver_id], riders_by_id[result.rider_id]
        )
        print(f"  {result.driver_id} -> {result.rider_id}  "
              f"(行車距離 = {result.cost:.0f} 公尺, 路線 {len(path)} 個點)")

    total_cost = sum(result.cost for result in results)
    print(f"\nTotal system cost: {total_cost:.0f} 公尺")
