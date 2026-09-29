"""即時共乘派單與最佳化配對系統 — 互動式演算法展示台 (Interactive Playground)。

執行方式：
    streamlit run app.py

操作流程：
1. 在側邊欄選擇點擊模式（新增司機 / 新增乘客 / 封鎖路口）。
2. 直接點擊地圖放置標記（司機=藍色、乘客=紅色、封路=黑色禁止圖示）。
3. 按「計算配對」：套用最新封路狀態，MatchingEngine 求全局最佳配對，
   MapRoutingAdapter 在繞過封鎖路口的前提下算出真實街道路徑並畫線。
4. 地圖視角（中心/縮放）跨 rerun 保留，點擊不會跳回預設位置。
5. 按「載入即時範例」：抓取臺北市公車此刻的 GPS 位置當司機，
   並在目前地圖畫面內隨機挑路口當乘客。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import folium
import streamlit as st
from streamlit_folium import st_folium

from live_data import (
    BusPosition,
    fetch_bus_positions,
    pick_live_drivers,
    pick_random_riders,
    view_radius_m,
)
from map_adapter import MapRoutingAdapter
from matching_engine import Driver, MatchingEngine, Rider

# --- 常數 ---
DEFAULT_MAP_CENTER = (25.0625, 121.5130)  # 大同區中心 (lat, lng)
DEFAULT_MAP_ZOOM = 14                     # 區級視野
MAP_WIDTH_PX = 1100
MAP_HEIGHT_PX = 550
# 點擊位置距最近路網節點超過此距離（公尺）就拒絕新增：
# 路網外的點會被 nearest_nodes 硬吸附到區界節點，路徑會畫到完全錯誤的位置
MAX_SNAP_DISTANCE_M = 250
# 每組配對輪流使用的路線顏色（重疊路段才分得出是哪一組）
PATH_COLORS = ["green", "purple", "orange", "darkred", "cadetblue"]
# 即時範例的人數：司機多於乘客，才看得出演算法在挑「派哪幾位司機」最划算
LIVE_DRIVER_COUNT = 6
LIVE_RIDER_COUNT = 4

# 預設範例：台北市各區地標（x=經度 lng, y=緯度 lat）
EXAMPLE_DRIVERS = [
    Driver(id="D1", x=121.5170, y=25.0478),  # 台北車站
    Driver(id="D2", x=121.5637, y=25.0408),  # 市政府（信義區）
    Driver(id="D3", x=121.5245, y=25.0880),  # 士林夜市
]
EXAMPLE_RIDERS = [
    Rider(id="R1", x=121.5070, y=25.0420),  # 西門町
    Rider(id="R2", x=121.5645, y=25.0339),  # 台北 101
]


@st.cache_resource(show_spinner="載入台北市路網中（首次需下載數分鐘，之後走快取）...")
def load_adapter() -> MapRoutingAdapter:
    """路網圖資只載入一次，跨 Streamlit rerun 重複使用。"""
    return MapRoutingAdapter()


@st.cache_data(ttl=30, show_spinner="抓取臺北市公車即時位置中...")
def load_bus_positions() -> List[BusPosition]:
    """公車位置快取 30 秒：短時間內重複按按鈕不必重新下載。"""
    return fetch_bus_positions()


def init_session_state() -> None:
    """初始化展示台的所有狀態（僅在第一次執行時生效）。"""
    defaults: Dict[str, Any] = {
        "drivers": [],            # List[Driver]
        "riders": [],             # List[Rider]
        "roadblocks": [],         # List[Tuple[float, float]]：封鎖路口 (lat, lng)
        "match_results": [],      # List[Dict]：配對結果與真實路徑
        "driver_labels": {},      # Dict[str, str]：司機 id → 公車車牌（即時範例）
        "last_processed_click": None,  # 已處理過的點擊座標（去重用）
        "map_center": DEFAULT_MAP_CENTER,  # 目前地圖中心 (lat, lng)，跨 rerun 保留
        "map_zoom": DEFAULT_MAP_ZOOM,      # 目前地圖縮放，跨 rerun 保留
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_matching(adapter: MapRoutingAdapter) -> None:
    """執行配對並把結果（含真實街道路徑）存入 session_state。"""
    drivers: List[Driver] = st.session_state.drivers
    riders: List[Rider] = st.session_state.riders

    # 先套用目前的封路狀態：在原圖副本上移除封鎖節點，
    # 之後的成本計算與路徑繪製都會自動繞過這些路口。
    adapter.apply_roadblocks(st.session_state.roadblocks)

    # 批次版成本函式：每位司機只跑一次單源 Dijkstra
    cost_fn = adapter.build_batch_cost_fn(drivers)
    engine = MatchingEngine(cost_fn=cost_fn)
    results = engine.match(drivers, riders)

    drivers_by_id = {d.id: d for d in drivers}
    riders_by_id = {r.id: r for r in riders}

    match_results: List[Dict[str, Any]] = []
    for result in results:
        driver = drivers_by_id[result.driver_id]
        rider = riders_by_id[result.rider_id]
        # 街道路徑是「最近路網節點」之間的串列，與標記實際位置有小段
        # 落差；頭尾補上司機/乘客座標，讓每條線都確實從藍色標記出發、
        # 結束在紅色標記，不會和相鄰路徑黏在一起。
        street_path = adapter.get_routing_path(driver, rider)
        full_path = [(driver.y, driver.x)] + street_path + [(rider.y, rider.x)]
        match_results.append(
            {
                "driver_id": result.driver_id,
                "rider_id": result.rider_id,
                "cost": result.cost,
                "path": full_path,
            }
        )
    st.session_state.match_results = match_results


def load_live_example(adapter: MapRoutingAdapter) -> None:
    """即時範例：目前地圖畫面內的公車即時位置當司機，隨機路口當乘客。

    取樣半徑依目前縮放計算，確保標記都落在畫面內；公車須在路網
    涵蓋範圍內（與點擊新增相同的吸附距離門檻），否則路徑會畫錯位置。
    """
    try:
        buses = load_bus_positions()
    except (OSError, ValueError) as error:
        # 網路斷線、逾時、檔案格式錯誤都只提示，不讓整頁崩潰
        st.error(f"無法取得公車即時資料（{error}），可以改按「載入預設範例」。")
        return
    if not buses:
        st.warning("目前沒有營運中的公車（可能是深夜收班時段），請改按「載入預設範例」。")
        return

    center_lat, center_lng = st.session_state.map_center
    radius = view_radius_m(center_lat, st.session_state.map_zoom, MAP_HEIGHT_PX)
    rng = random.Random()
    drivers, bus_ids = pick_live_drivers(
        buses, center_lng, center_lat, radius, LIVE_DRIVER_COUNT, rng,
        is_on_network=lambda x, y: adapter.snap_distance_m(x, y) <= MAX_SNAP_DISTANCE_M,
    )
    if not drivers:
        st.warning(
            f"目前地圖中心 {radius:,.0f} 公尺內沒有營運中的公車，"
            "請把地圖移到台北市區再試。"
        )
        return

    # 乘客從路網節點（路口）取樣，保證落在街道上
    nodes = [(data["x"], data["y"]) for _, data in adapter.graph.nodes(data=True)]
    riders = pick_random_riders(nodes, center_lng, center_lat, radius, LIVE_RIDER_COUNT, rng)

    st.session_state.drivers = drivers
    st.session_state.riders = riders
    st.session_state.driver_labels = bus_ids
    st.session_state.match_results = []
    newest = max(bus.data_time for bus in buses)
    st.success(
        f"已載入 {len(drivers)} 輛公車的即時位置（資料時間 {newest:%H:%M:%S}）"
        f"與 {len(riders)} 位隨機乘客，按「計算配對」開始派單。"
    )


def render_map() -> folium.Map:
    """依目前狀態畫地圖：藍=司機、紅=乘客、綠線=配對的真實路徑。"""
    # 初始視角用 session_state 保留的中心/縮放，st_folium 的 center/zoom
    # 參數會再強化一次，確保每次 rerun 都停在使用者最後瀏覽的位置。
    fmap = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        # CARTO 的淺色底圖改成需要 API key（放大後整片都是 API KEY REQUIRED 浮水印），
        # 改用 OpenStreetMap 官方圖磚：免金鑰，各級縮放都有真實圖資
        tiles="OpenStreetMap",
    )

    for driver in st.session_state.drivers:
        bus_id = st.session_state.driver_labels.get(driver.id)
        folium.Marker(
            location=(driver.y, driver.x),  # folium 吃 (lat, lng)
            tooltip=f"{driver.id}（司機・公車 {bus_id}）" if bus_id else f"{driver.id}（司機）",
            icon=folium.Icon(color="blue", icon="car", prefix="fa"),
        ).add_to(fmap)

    for rider in st.session_state.riders:
        folium.Marker(
            location=(rider.y, rider.x),
            tooltip=f"{rider.id}（乘客）",
            icon=folium.Icon(color="red", icon="user", prefix="fa"),
        ).add_to(fmap)

    for lat, lng in st.session_state.roadblocks:
        folium.Marker(
            location=(lat, lng),  # 封路座標本來就以 (lat, lng) 儲存
            tooltip="🚧 封鎖路口",
            icon=folium.Icon(color="black", icon="ban", prefix="fa"),
        ).add_to(fmap)

    # 每一組配對各自實例化一個獨立的 PolyLine（絕不共用座標串列），
    # 並輪流配色，重疊路段才看得出分屬哪一組
    for index, match in enumerate(st.session_state.match_results):
        if match["path"]:
            folium.PolyLine(
                locations=match["path"],
                color=PATH_COLORS[index % len(PATH_COLORS)],
                weight=5,
                opacity=0.85,
                tooltip=(
                    f"{match['driver_id']} → {match['rider_id']}"
                    f"（{match['cost']:.0f} 公尺）"
                ),
            ).add_to(fmap)

    return fmap


# --- 版面配置 ---
st.set_page_config(page_title="共乘配對演算法展示台", layout="wide")
st.title("🧪 共乘配對演算法展示台 — 台北市")

init_session_state()
adapter = load_adapter()

with st.sidebar:
    st.header("操作面板")
    click_mode = st.radio(
        "點擊模式（點地圖會新增…）",
        ["新增司機", "新增乘客", "🚧 封鎖路口"],
    )
    run_clicked = st.button("計算配對 (Run Matching)", type="primary",
                            use_container_width=True)
    clear_clicked = st.button("清除所有資料 (Clear All)", use_container_width=True)
    example_clicked = st.button("載入預設範例 (Load Example)",
                                use_container_width=True)
    live_clicked = st.button("載入即時範例 (Live Example)",
                             use_container_width=True)
    st.caption("提示：請在台北市範圍內點擊新增標記或封路，"
               "再按「計算配對」畫出真實行車路線；地圖會保持目前視角。")
    st.caption("即時範例：叫車平台的司機位置沒有公開資料，這裡改用"
               "[臺北市資料大平臺](https://data.taipei/)的公車即時 GPS "
               "當作司機位置，乘客則在目前地圖畫面內隨機產生。")

# --- 側邊欄按鈕處理（在渲染主畫面前先改完狀態）---
if clear_clicked:
    st.session_state.drivers = []
    st.session_state.riders = []
    st.session_state.roadblocks = []
    st.session_state.match_results = []
    st.session_state.driver_labels = {}

if example_clicked:
    # list(...) 複製一份，避免使用者點擊時改到範例常數
    st.session_state.drivers = list(EXAMPLE_DRIVERS)
    st.session_state.riders = list(EXAMPLE_RIDERS)
    st.session_state.match_results = []
    st.session_state.driver_labels = {}

if live_clicked:
    load_live_example(adapter)

# --- 手動配對：按下「計算配對」才執行（內部會先套用最新封路狀態）---
if run_clicked:
    if st.session_state.drivers and st.session_state.riders:
        run_matching(adapter)
    else:
        st.warning("請先在地圖上加入至少一名司機與一名乘客，再計算配對。")

# --- 主畫面：數量指標 ---
col_driver, col_rider, col_block, col_match = st.columns(4)
col_driver.metric("司機數量", len(st.session_state.drivers))
col_rider.metric("乘客數量", len(st.session_state.riders))
col_block.metric("封鎖路口", len(st.session_state.roadblocks))
col_match.metric("配對成功", len(st.session_state.match_results))

# --- 互動式地圖 ---
map_state = st_folium(
    render_map(),
    key="playground_map",
    width=MAP_WIDTH_PX,
    height=MAP_HEIGHT_PX,
    center=st.session_state.map_center,  # 用保留的視角渲染，避免每次 rerun 跳回預設
    zoom=st.session_state.map_zoom,
    returned_objects=["last_clicked", "center", "zoom"],
)
map_state = map_state or {}

# --- 視角保存：把使用者最後瀏覽的中心/縮放寫回 session_state ---
# 必須在點擊觸發 rerun 之前先存，下次 rerun 地圖才會停在原位不亂跳。
returned_center = map_state.get("center")
if returned_center:
    st.session_state.map_center = (returned_center["lat"], returned_center["lng"])
returned_zoom = map_state.get("zoom")
if returned_zoom is not None:
    st.session_state.map_zoom = returned_zoom

# --- 點擊事件處理：依模式新增司機或乘客 ---
clicked = map_state.get("last_clicked")
if clicked:
    # st_folium 在每次 rerun 都會回傳同一筆 last_clicked，
    # 必須與已處理過的座標比對去重，否則會無限重複新增。
    point = (round(clicked["lat"], 6), round(clicked["lng"], 6))
    if point != st.session_state.last_processed_click:
        st.session_state.last_processed_click = point
        lat, lng = point
        # 路網涵蓋範圍檢查：太遠的點會被吸附到區界節點，路徑會畫錯位置
        snap_distance = adapter.snap_distance_m(lng, lat)
        if snap_distance > MAX_SNAP_DISTANCE_M:
            st.warning(
                f"點擊位置距離台北市路網約 {snap_distance:,.0f} 公尺，"
                "超出涵蓋範圍，未新增標記——請點擊台北市內的街道附近。"
            )
        else:
            if click_mode == "新增司機":
                new_id = f"D{len(st.session_state.drivers) + 1}"
                st.session_state.drivers.append(Driver(id=new_id, x=lng, y=lat))
            elif click_mode == "新增乘客":
                new_id = f"R{len(st.session_state.riders) + 1}"
                st.session_state.riders.append(Rider(id=new_id, x=lng, y=lat))
            else:  # 🚧 封鎖路口
                st.session_state.roadblocks.append((lat, lng))
            # 佈局或路網改變後，舊的配對結果已失效
            st.session_state.match_results = []
            st.rerun()

# --- 配對結果明細 ---
if st.session_state.match_results:
    st.subheader("配對結果")
    st.table(
        [
            {
                "司機": match["driver_id"],
                "乘客": match["rider_id"],
                "行車距離（公尺）": f"{match['cost']:.0f}",
                "路徑節點數": len(match["path"]),
            }
            for match in st.session_state.match_results
        ]
    )
    total_cost = sum(match["cost"] for match in st.session_state.match_results)
    st.caption(f"系統總成本：{total_cost:.0f} 公尺（全局最低總成本配對）")
