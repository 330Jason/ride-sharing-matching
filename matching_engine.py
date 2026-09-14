"""即時共乘派單與最佳化配對系統 — 第一階段核心模組。

MatchingEngine 是純粹的演算法黑盒子：
- 僅在 2D 網格上運作，不依賴任何地圖或網路套件。
- 成本以曼哈頓距離模擬行車成本。
- 透過匈牙利演算法 (scipy.optimize.linear_sum_assignment) 求全局最低總成本配對。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Driver:
    """司機：id 與其在 2D 網格上的座標。"""

    id: str
    x: float
    y: float


@dataclass
class Rider:
    """乘客：id、上車點座標 (x, y) 與目的地座標 (dest_x, dest_y)。

    dest_x / dest_y 未提供時預設為上車點（原地）——
    純演算法層只用上車點計算配對成本，不依賴目的地；
    這個回退讓既有呼叫端不受影響，也避免目的地被誤設為 (0, 0)。
    __post_init__ 之後兩者保證為 float。
    """

    id: str
    x: float
    y: float
    dest_x: Optional[float] = None
    dest_y: Optional[float] = None

    def __post_init__(self) -> None:
        if self.dest_x is None:
            self.dest_x = self.x
        if self.dest_y is None:
            self.dest_y = self.y


@dataclass
class MatchResult:
    """單筆配對結果：哪位司機接哪位乘客，以及該趟的成本。"""

    driver_id: str
    rider_id: str
    cost: float


def manhattan_distance(driver: Driver, rider: Rider) -> float:
    """預設成本函式：以曼哈頓距離模擬行車成本。"""
    return abs(driver.x - rider.x) + abs(driver.y - rider.y)


class MatchingEngine:
    """司機—乘客二部圖配對引擎（無狀態，可重複呼叫）。

    cost_fn 可注入自訂成本函式（預設為曼哈頓距離）。
    成本為 float('inf') 代表該司機無法到達該乘客，
    配對時絕不會產生落在不可達組合上的結果。
    """

    def __init__(
        self, cost_fn: Callable[[Driver, Rider], float] = manhattan_distance
    ) -> None:
        self.cost_fn = cost_fn

    def build_cost_matrix(
        self, drivers: List[Driver], riders: List[Rider]
    ) -> np.ndarray:
        """計算成本矩陣，shape 為 (len(drivers), len(riders))。

        matrix[i][j] = cost_fn(第 i 位司機, 第 j 位乘客)。
        司機與乘客數量不必相等，允許非方陣。
        """
        matrix = np.zeros((len(drivers), len(riders)), dtype=float)
        for i, driver in enumerate(drivers):
            for j, rider in enumerate(riders):
                matrix[i, j] = self.cost_fn(driver, rider)
        return matrix

    def match(
        self, drivers: List[Driver], riders: List[Rider]
    ) -> List[MatchResult]:
        """以匈牙利演算法求全局總成本最低的司機—乘客配對。

        linear_sum_assignment 原生支援非方陣：
        配對數量 = min(len(drivers), len(riders))，多出來的一方不被配對。
        任一方為空時回傳空清單。
        成本為 inf（不可達）的組合不會出現在結果中；
        若某一方完全不可達，其餘參與者仍會正常配對。
        """
        if not drivers or not riders:
            return []

        cost_matrix = self.build_cost_matrix(drivers, riders)

        # scipy 不接受 inf，改用「可達性遮罩 + 哨兵值」處理不可達組合：
        # 哨兵值嚴格大於任何完整可行配對的總成本，演算法只會在被迫時選到，
        # 求解後再把落在不可達邊上的配對濾掉。
        feasible = np.isfinite(cost_matrix)
        if not feasible.any():
            return []
        sentinel = cost_matrix[feasible].max() * min(cost_matrix.shape) + 1.0
        solvable_matrix = np.where(feasible, cost_matrix, sentinel)

        row_indices, col_indices = linear_sum_assignment(solvable_matrix)

        return [
            MatchResult(
                driver_id=drivers[i].id,
                rider_id=riders[j].id,
                cost=float(cost_matrix[i, j]),
            )
            for i, j in zip(row_indices, col_indices)
            if feasible[i, j]
        ]


if __name__ == "__main__":
    random.seed(42)  # 固定種子，方便重現結果

    # 1. 隨機生成 5 個 Driver 與 3 個 Rider（座標落在 0~100 的網格）
    drivers = [
        Driver(id=f"D{i + 1}", x=round(random.uniform(0, 100), 1), y=round(random.uniform(0, 100), 1))
        for i in range(5)
    ]
    riders = [
        Rider(id=f"R{i + 1}", x=round(random.uniform(0, 100), 1), y=round(random.uniform(0, 100), 1))
        for i in range(3)
    ]

    print("=== Drivers ===")
    for d in drivers:
        print(f"  {d.id}: ({d.x}, {d.y})")
    print("=== Riders ===")
    for r in riders:
        print(f"  {r.id}: ({r.x}, {r.y})")

    # 2. 實例化引擎並執行配對
    engine = MatchingEngine()

    cost_matrix = engine.build_cost_matrix(drivers, riders)
    print("\n=== Cost Matrix (rows=Drivers, cols=Riders) ===")
    header = "      " + "".join(f"{r.id:>10}" for r in riders)
    print(header)
    for d, row in zip(drivers, cost_matrix):
        print(f"  {d.id:>3} " + "".join(f"{cost:>10.1f}" for cost in row))

    results = engine.match(drivers, riders)

    # 3. 印出配對結果與系統總成本
    print("\n=== Match Results ===")
    for result in results:
        print(f"  {result.driver_id} -> {result.rider_id}  (cost = {result.cost:.1f})")

    total_cost = sum(result.cost for result in results)
    print(f"\nTotal system cost: {total_cost:.1f}")
