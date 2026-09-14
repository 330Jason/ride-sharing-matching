"""MatchingEngine 的單元測試。

涵蓋範圍：
- 成本矩陣的形狀與曼哈頓距離數值（含非方陣）
- 空輸入、1 對 1、司機/乘客數量不對等的邊界條件
- 全局最佳性（反貪婪案例 + 與暴力枚舉解對照）
- 不可達組合（成本為 inf）：跳過不可達邊、其他人仍正確配對
"""

from __future__ import annotations

import math
import random
from itertools import permutations
from typing import List, Set, Tuple

import numpy as np
import pytest

from matching_engine import (
    Driver,
    MatchingEngine,
    Rider,
    manhattan_distance,
)

# ---------------------------------------------------------------------------
# 測試輔助工具
# ---------------------------------------------------------------------------


def make_drivers(*coords: Tuple[float, float]) -> List[Driver]:
    """依座標順序產生 D1, D2, ... 的司機清單。"""
    return [Driver(id=f"D{i + 1}", x=x, y=y) for i, (x, y) in enumerate(coords)]


def make_riders(*coords: Tuple[float, float]) -> List[Rider]:
    """依座標順序產生 R1, R2, ... 的乘客清單。"""
    return [Rider(id=f"R{i + 1}", x=x, y=y) for i, (x, y) in enumerate(coords)]


def matched_pairs(results) -> Set[Tuple[str, str]]:
    """把配對結果轉成 {(driver_id, rider_id)} 集合，方便斷言。"""
    return {(r.driver_id, r.rider_id) for r in results}


def brute_force_total(matrix: np.ndarray) -> float:
    """枚舉所有可能配對組合，回傳最低總成本（僅適用於小矩陣）。"""
    n_drivers, n_riders = matrix.shape
    if n_drivers <= n_riders:
        return min(
            sum(matrix[i, cols[i]] for i in range(n_drivers))
            for cols in permutations(range(n_riders), n_drivers)
        )
    return min(
        sum(matrix[rows[j], j] for j in range(n_riders))
        for rows in permutations(range(n_drivers), n_riders)
    )


def engine_with_blocked(blocked: Set[Tuple[str, str]]) -> MatchingEngine:
    """回傳一個把 blocked 中 (driver_id, rider_id) 視為不可達的引擎。"""

    def cost_fn(driver: Driver, rider: Rider) -> float:
        if (driver.id, rider.id) in blocked:
            return math.inf
        return manhattan_distance(driver, rider)

    return MatchingEngine(cost_fn=cost_fn)


@pytest.fixture
def engine() -> MatchingEngine:
    """預設引擎：曼哈頓距離成本。"""
    return MatchingEngine()


# ---------------------------------------------------------------------------
# 成本矩陣
# ---------------------------------------------------------------------------


class TestCostMatrix:
    def test_non_square_shape_more_riders(self, engine: MatchingEngine) -> None:
        matrix = engine.build_cost_matrix(
            make_drivers((0, 0), (1, 1)),
            make_riders((0, 0), (1, 1), (2, 2)),
        )
        assert matrix.shape == (2, 3)

    def test_non_square_shape_more_drivers(self, engine: MatchingEngine) -> None:
        matrix = engine.build_cost_matrix(
            make_drivers((0, 0), (1, 1), (2, 2), (3, 3)),
            make_riders((0, 0), (1, 1)),
        )
        assert matrix.shape == (4, 2)

    def test_manhattan_values(self, engine: MatchingEngine) -> None:
        matrix = engine.build_cost_matrix(
            make_drivers((0.0, 0.0), (-1.0, -2.0)),
            make_riders((3.0, 4.0)),
        )
        assert matrix[0, 0] == pytest.approx(7.0)  # |0-3| + |0-4|
        assert matrix[1, 0] == pytest.approx(10.0)  # |-1-3| + |-2-4|

    def test_custom_cost_fn_is_used(self) -> None:
        """注入的成本函式（含 inf）要能反映在矩陣中。"""
        eng = engine_with_blocked({("D1", "R1")})
        matrix = eng.build_cost_matrix(
            make_drivers((0, 0)), make_riders((3, 4), (1, 0))
        )
        assert np.isinf(matrix[0, 0])
        assert matrix[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rider 目的地欄位（Phase 4）
# ---------------------------------------------------------------------------


class TestRiderDestination:
    def test_destination_defaults_to_pickup(self) -> None:
        """未指定目的地時，dest_x/dest_y 回退為上車點（不是 0,0）。"""
        rider = Rider(id="R1", x=1.5, y=2.5)
        assert (rider.dest_x, rider.dest_y) == (1.5, 2.5)

    def test_explicit_destination_preserved(self) -> None:
        rider = Rider(id="R1", x=1.0, y=2.0, dest_x=3.0, dest_y=4.0)
        assert (rider.dest_x, rider.dest_y) == (3.0, 4.0)

    def test_matching_cost_ignores_destination(self, engine: MatchingEngine) -> None:
        """配對成本只看上車點：目的地再遠都不影響成本矩陣。"""
        riders = [Rider(id="R1", x=3.0, y=4.0, dest_x=999.0, dest_y=999.0)]
        matrix = engine.build_cost_matrix(make_drivers((0, 0)), riders)
        assert matrix[0, 0] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# match() 邊界條件
# ---------------------------------------------------------------------------


class TestMatchEdgeCases:
    def test_empty_drivers(self, engine: MatchingEngine) -> None:
        assert engine.match([], make_riders((0, 0))) == []

    def test_empty_riders(self, engine: MatchingEngine) -> None:
        assert engine.match(make_drivers((0, 0)), []) == []

    def test_both_empty(self, engine: MatchingEngine) -> None:
        assert engine.match([], []) == []

    def test_one_to_one(self, engine: MatchingEngine) -> None:
        results = engine.match(make_drivers((0, 0)), make_riders((3, 4)))
        assert matched_pairs(results) == {("D1", "R1")}
        assert results[0].cost == pytest.approx(7.0)

    def test_more_drivers_than_riders(self, engine: MatchingEngine) -> None:
        """司機過剩：配對數 = 乘客數，每位乘客恰好被接一次。"""
        results = engine.match(
            make_drivers((0, 0), (10, 0), (20, 0), (30, 0)),
            make_riders((1, 0), (29, 0)),
        )
        assert len(results) == 2
        assert {r.rider_id for r in results} == {"R1", "R2"}
        # 同一位司機不得重複出現
        driver_ids = [r.driver_id for r in results]
        assert len(driver_ids) == len(set(driver_ids))

    def test_more_riders_than_drivers(self, engine: MatchingEngine) -> None:
        """乘客過剩：配對數 = 司機數，且應選成本最低的乘客組合。"""
        results = engine.match(
            make_drivers((0, 0), (100, 100)),
            make_riders((0, 1), (50, 50), (100, 101)),
        )
        assert matched_pairs(results) == {("D1", "R1"), ("D2", "R3")}
        assert sum(r.cost for r in results) == pytest.approx(2.0)

    def test_result_cost_equals_manhattan(self, engine: MatchingEngine) -> None:
        drivers = make_drivers((2.5, 3.5), (8.0, 1.0))
        riders = make_riders((4.0, 4.0), (7.0, 0.5))
        by_id = {d.id: d for d in drivers} | {r.id: r for r in riders}
        for result in engine.match(drivers, riders):
            expected = manhattan_distance(
                by_id[result.driver_id], by_id[result.rider_id]
            )
            assert result.cost == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 全局最佳性
# ---------------------------------------------------------------------------


class TestGlobalOptimality:
    def test_avoids_greedy_trap(self, engine: MatchingEngine) -> None:
        """貪婪法會先搶最便宜的邊 (D2→R1=3) 而導致總成本 17；
        匈牙利演算法應找到全局最佳 D1→R1 + D2→R2 = 11。"""
        results = engine.match(
            make_drivers((0, 0), (0, 8)),
            make_riders((0, 5), (0, 14)),
        )
        assert matched_pairs(results) == {("D1", "R1"), ("D2", "R2")}
        assert sum(r.cost for r in results) == pytest.approx(11.0)

    @pytest.mark.parametrize("n_drivers,n_riders", [(5, 3), (3, 5), (4, 4)])
    def test_total_cost_matches_brute_force(
        self, engine: MatchingEngine, n_drivers: int, n_riders: int
    ) -> None:
        """隨機網格上的引擎總成本必須等於暴力枚舉出的最低總成本。"""
        rng = random.Random(7)
        drivers = make_drivers(
            *((rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n_drivers))
        )
        riders = make_riders(
            *((rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n_riders))
        )

        results = engine.match(drivers, riders)
        total = sum(r.cost for r in results)

        matrix = engine.build_cost_matrix(drivers, riders)
        assert len(results) == min(n_drivers, n_riders)
        assert total == pytest.approx(brute_force_total(matrix))


# ---------------------------------------------------------------------------
# 不可達（成本 inf）
# ---------------------------------------------------------------------------


class TestUnreachable:
    def test_blocked_pair_forces_alternative_assignment(self) -> None:
        """D1→R1 不可達時，引擎應改採交叉配對，且結果不含不可達邊。"""
        eng = engine_with_blocked({("D1", "R1")})
        results = eng.match(
            make_drivers((0, 0), (10, 10)),
            make_riders((0, 1), (10, 11)),
        )
        assert matched_pairs(results) == {("D1", "R2"), ("D2", "R1")}
        assert all(math.isfinite(r.cost) for r in results)

    def test_fully_blocked_driver_leaves_others_matched(self) -> None:
        """D2 對所有乘客都不可達：D2 落單，其餘司機仍取得全局最佳配對。"""
        eng = engine_with_blocked({("D2", "R1"), ("D2", "R2"), ("D2", "R3")})
        results = eng.match(
            make_drivers((0, 0), (50, 50), (10, 10)),
            make_riders((1, 0), (11, 10), (40, 40)),
        )
        assert matched_pairs(results) == {("D1", "R1"), ("D3", "R2")}
        assert "D2" not in {r.driver_id for r in results}
        assert sum(r.cost for r in results) == pytest.approx(2.0)

    def test_blocked_pair_never_appears_even_when_forced(self) -> None:
        """兩位司機都到不了 R1：R1 落單，R2 由較近的司機接走。"""
        eng = engine_with_blocked({("D1", "R1"), ("D2", "R1")})
        results = eng.match(
            make_drivers((0, 0), (5, 5)),
            make_riders((100, 100), (6, 5)),
        )
        assert matched_pairs(results) == {("D2", "R2")}

    def test_all_pairs_blocked_returns_empty(self) -> None:
        eng = engine_with_blocked(
            {("D1", "R1"), ("D1", "R2"), ("D2", "R1"), ("D2", "R2")}
        )
        results = eng.match(
            make_drivers((0, 0), (1, 1)),
            make_riders((2, 2), (3, 3)),
        )
        assert results == []
