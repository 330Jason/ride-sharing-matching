"""即時共乘派單與最佳化配對系統 — 第三階段：事件驅動批量派單模擬器。

RideSharingSimulator 組合前兩階段的模組：
- MapRoutingAdapter：提供真實路網行車距離（批次版單源 Dijkstra）。
- MatchingEngine：求每個批次的全局最低總成本配對。

時間以「秒」為單位推進（tick），每 BATCH_INTERVAL 秒觸發一次批量派單。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from map_adapter import MapRoutingAdapter
from matching_engine import Driver, MatchingEngine, Rider


class RideSharingSimulator:
    """事件驅動模擬器：管理司機/乘客狀態，並按時間窗口批量派單。"""

    BATCH_INTERVAL: int = 5  # 派單時間窗口：每 5 秒一個批次
    AVERAGE_SPEED_MPS: float = 30 / 3.6  # 平均時速 30 km/h ≈ 8.33 m/s

    def __init__(
        self,
        adapter: MapRoutingAdapter,
        drivers: Optional[List[Driver]] = None,
        log_fn: Callable[[str], None] = print,
    ) -> None:
        """log_fn：事件 Log 的輸出函式，預設印到終端機；
        儀表板等前端可注入自訂函式收集事件（如 list.append）。"""
        self.adapter = adapter
        self._log_fn = log_fn
        self.current_time: int = 0
        self.idle_drivers: List[Driver] = list(drivers) if drivers else []
        self.waiting_riders: List[Rider] = []
        # 進行中的旅程：{"driver", "rider_id", "distance_m",
        #               "completes_at", "destination"}
        self.active_trips: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 對外操作
    # ------------------------------------------------------------------

    def add_driver(self, driver: Driver) -> None:
        """新司機上線，加入閒置清單。"""
        self.idle_drivers.append(driver)

    def add_rider(self, rider: Rider) -> None:
        """乘客動態叫車：進入等待佇列，待下一個派單批次處理。"""
        self.waiting_riders.append(rider)
        self._log(
            f"Rider {rider.id} requested a ride at ({rider.x:.4f}, {rider.y:.4f})"
        )

    def tick(self) -> None:
        """時間推進 1 秒：先結算完成的旅程，再視時間窗口觸發批量派單。"""
        self.current_time += 1
        self._complete_trips()
        if (
            self.current_time % self.BATCH_INTERVAL == 0
            and self.waiting_riders
            and self.idle_drivers
        ):
            self._dispatch_batch()

    # ------------------------------------------------------------------
    # 內部邏輯
    # ------------------------------------------------------------------

    def _complete_trips(self) -> None:
        """把已到完成時間的旅程結算掉：司機移動到終點並回到閒置清單。"""
        still_active: List[Dict[str, Any]] = []
        for trip in self.active_trips:
            if trip["completes_at"] <= self.current_time:
                driver: Driver = trip["driver"]
                driver.x, driver.y = trip["destination"]
                self.idle_drivers.append(driver)
                self._log(
                    f"Trip completed: {driver.id} finished trip with "
                    f"{trip['rider_id']}, now idle at ({driver.x:.4f}, {driver.y:.4f})"
                )
            else:
                still_active.append(trip)
        self.active_trips = still_active

    def _dispatch_batch(self) -> None:
        """批量派單：用批次版成本函式跑一次全局最佳配對並更新狀態。"""
        self._log(
            f"Batch dispatch triggered: {len(self.idle_drivers)} idle driver(s), "
            f"{len(self.waiting_riders)} waiting rider(s)"
        )

        # 批次版成本函式：每位司機只跑一次單源 Dijkstra（見 MapRoutingAdapter）
        cost_fn = self.adapter.build_batch_cost_fn(self.idle_drivers)
        # MatchingEngine 無狀態、建構成本為零，每批次以當批 cost_fn 建一個
        engine = MatchingEngine(cost_fn=cost_fn)
        results = engine.match(self.idle_drivers, self.waiting_riders)

        drivers_by_id = {d.id: d for d in self.idle_drivers}
        riders_by_id = {r.id: r for r in self.waiting_riders}

        for result in results:
            driver = drivers_by_id[result.driver_id]
            rider = riders_by_id[result.rider_id]

            # 旅程 = 接客段（配對成本：司機→上車點）+ 載客段（上車點→目的地）
            pickup_distance = result.cost
            dropoff_distance = self.adapter.shortest_distance(
                rider.x, rider.y, rider.dest_x, rider.dest_y
            )
            if not math.isfinite(dropoff_distance):
                # 目的地在路網上不可達：退而求其次，旅程終點視為上車點
                dropoff_distance = 0.0
                destination = (rider.x, rider.y)
            else:
                destination = (rider.dest_x, rider.dest_y)

            # 預計完成時間 = 總距離 / 平均速度（至少 1 秒）
            total_distance = pickup_distance + dropoff_distance
            duration = max(1, math.ceil(total_distance / self.AVERAGE_SPEED_MPS))
            completes_at = self.current_time + duration

            self.active_trips.append(
                {
                    "driver": driver,
                    "rider_id": rider.id,
                    "distance_m": total_distance,
                    "completes_at": completes_at,
                    "destination": destination,
                }
            )
            self.idle_drivers.remove(driver)
            self.waiting_riders.remove(rider)

            self._log(
                f"Matched {driver.id} to {rider.id} "
                f"(pickup {pickup_distance:.0f}m + trip {dropoff_distance:.0f}m), "
                f"ETA: {duration}s (completes at {self._format_time(completes_at)})"
            )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _format_time(seconds: int) -> str:
        minutes, secs = divmod(seconds, 60)
        return f"{minutes:02d}:{secs:02d}"

    def _log(self, message: str) -> None:
        self._log_fn(f"[Time: {self._format_time(self.current_time)}] {message}")


if __name__ == "__main__":
    # 1. 載入台北市路網並初始化 5 名閒置司機——大同區一帶（x=經度 lng, y=緯度 lat）
    adapter = MapRoutingAdapter()
    initial_drivers = [
        Driver(id="D1", x=121.5155, y=25.0495),  # 台北車站北側
        Driver(id="D2", x=121.5128, y=25.0630),  # 大橋頭捷運站
        Driver(id="D3", x=121.5100, y=25.0555),  # 迪化街商圈
        Driver(id="D4", x=121.5155, y=25.0565),  # 寧夏夜市
        Driver(id="D5", x=121.5110, y=25.0660),  # 延平北路三段
    ]
    simulator = RideSharingSimulator(adapter, drivers=initial_drivers)

    # 2. 預先排程的動態叫車事件：模擬秒數 -> 該秒加入的乘客
    rider_schedule: Dict[int, List[Rider]] = {
        2: [
            # 永樂市場 -> 大橋頭捷運站（含目的地）
            Rider(id="R1", x=121.5103, y=25.0547, dest_x=121.5128, dest_y=25.0630),
            # 建成公園 -> 台北車站北側（含目的地）
            Rider(id="R2", x=121.5167, y=25.0568, dest_x=121.5155, dest_y=25.0495),
        ],
        12: [
            # 朝陽公園（未指定目的地：回退為上車點）
            Rider(id="R3", x=121.5135, y=25.0530),
        ],
        27: [
            # 大橋頭北側 -> 迪化街商圈（含目的地）
            Rider(id="R4", x=121.5120, y=25.0645, dest_x=121.5100, dest_y=25.0555),
            # 台北車站周邊（未指定目的地：回退為上車點）
            Rider(id="R5", x=121.5150, y=25.0500),
        ],
    }

    # 3. 模擬 4 分鐘：每秒 tick 一次，並在排程時間點動態加入乘客
    #   （旅程含載客段後較長，240 秒才看得到完成與司機移動到目的地）
    print("=== Simulation start: 240 seconds, batch every "
          f"{RideSharingSimulator.BATCH_INTERVAL}s ===")
    for _ in range(240):
        simulator.tick()
        for new_rider in rider_schedule.get(simulator.current_time, []):
            simulator.add_rider(new_rider)

    # 4. 模擬結束後的狀態總結
    print(f"\n=== Simulation summary at {simulator._format_time(simulator.current_time)} ===")
    print(f"  Idle drivers   : {[d.id for d in simulator.idle_drivers]}")
    print(f"  Waiting riders : {[r.id for r in simulator.waiting_riders]}")
    print(
        "  Active trips   : "
        + str(
            [
                (t["driver"].id, t["rider_id"], f"ETA {simulator._format_time(t['completes_at'])}")
                for t in simulator.active_trips
            ]
        )
    )
