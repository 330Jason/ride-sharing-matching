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
from typing import Callable, Dict, List, Tuple

import networkx as nx
import osmnx as ox

from matching_engine import Driver, Rider


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
        """司機到乘客上車點的真實街道路徑，回傳沿途節點座標串列。

        座標格式為 [(lat, lng), ...]——刻意採用 folium/leaflet 的
        (緯度, 經度) 順序，前端可直接餵給 folium.PolyLine 繪製。
        走 working_graph，因此會自動繞過已封鎖的路口；
        路網不通或起訖點被封鎖時回傳空串列。
        """
        source = self._nearest_node(driver.x, driver.y)
        target = self._nearest_node(rider.x, rider.y)
        try:
            node_ids = nx.shortest_path(
                self.working_graph, source, target, weight="length"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        return [
            (self.working_graph.nodes[node]["y"],
             self.working_graph.nodes[node]["x"])
            for node in node_ids
        ]

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

    print("\n=== Match Results ===")
    for result in results:
        print(f"  {result.driver_id} -> {result.rider_id}  "
              f"(行車距離 = {result.cost:.0f} 公尺)")

    total_cost = sum(result.cost for result in results)
    print(f"\nTotal system cost: {total_cost:.0f} 公尺")
