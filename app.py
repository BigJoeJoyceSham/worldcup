"""World Cup 2026 predictions dashboard.

Five views over the friends-league prediction pool:
  Live Table  - leaderboard with movement + form
  Title Race  - animated cumulative-points race + bump (rank) chart
  Heatmap     - players x matches, coloured by points won
  Match       - every player's prediction vs the actual result, one game
  Player      - a single player's profile, tendencies and breakdown

Data comes from the shared Google Sheet via data_loader.load_long(). A live
score feed (API / scrape) can later replace the Results half of that loader
with no change here.
"""
from __future__ import annotations

import os
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_loader as dl
import extras
import cards as cardui

try:
    from streamlit_echarts import st_echarts
    _HAS_ECHARTS = True
except Exception:  # pragma: no cover - fall back to Plotly
    _HAS_ECHARTS = False

st.set_page_config(page_title="World Cup 2026 Predictions", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed",
                   # Blank out the menu entries that would otherwise deep-link to
                   # the GitHub repo ("Report a bug" / "View source").
                   menu_items={"Get Help": None, "Report a bug": None,
                               "About": "World Cup 2026 Predictions"})

# Belt-and-suspenders: hide the top-right toolbar (Deploy button + the source/
# GitHub badge Streamlit Cloud injects) via CSS, on top of toolbarMode=minimal.
# Also force the sidebar permanently collapsed and not openable — the dev widgets
# inside it still run server-side with their defaults (live source, API on), so
# the app works; viewers just can't reach the controls.
st.markdown(
    "<style>"
    "[data-testid='stToolbar']{display:none !important;}"
    "[data-testid='stStatusWidget']{display:none !important;}"
    "[data-testid='stSidebar']{display:none !important;}"
    "[data-testid='stSidebarCollapsedControl']{display:none !important;}"
    "[data-testid='collapsedControl']{display:none !important;}"
    "</style>",
    unsafe_allow_html=True)

# Navy / accent palette, plus one stable colour per player for every chart.
C = {"navy": "#1E3A5F", "red": "#DC2626", "amber": "#F59E0B",
     "green": "#16A34A", "grey": "#9CA3AF", "ink": "#1c1919"}
PALETTE = px.colors.qualitative.Bold
SNAPSHOT = os.path.join(os.path.dirname(__file__), "wc2026.xlsx")

extras.setup_plotly_theme(C, PALETTE)  # one house style for every chart


LIVE_TTL = 120  # seconds — most stale data may be while matches are in play
LIVE_END_HOUR = 3  # ET hour the day's live window closes (covers late games)
# Fixture clocks are stored/displayed in UK local time; match *days* are bucketed
# in New York time (single source of truth in data_loader), so a single match
# day can straddle two UK calendar days.
DISPLAY_TZ = dl.DISPLAY_TZ
MATCHDAY_TZ = dl.MATCHDAY_TZ


@st.cache_data(ttl=300, show_spinner=False)
def workbook_bytes(source: str) -> bytes:
    """Download the predictions workbook every 5 minutes and share the raw bytes
    across every loader. Short TTL so predictions entered in the Sheet on a
    matchday morning surface promptly (people submit picks right up to kickoff);
    live scores still come from the API, not a re-fetch."""
    return dl._read_workbook("live" if source == "live" else SNAPSHOT)


@st.cache_data(ttl=None, show_spinner="Pulling latest results…")
def get_data(source: str, use_api: bool, bucket: str) -> pd.DataFrame:
    """Load the tidy long table. Predictions/fixtures come from the Sheet;
    actual scores from the live API when enabled. Falls back to the bundled
    snapshot if the Sheet pull fails (offline / Google hiccup).

    ``bucket`` is a cache key, not used in the body: it changes every 2 minutes
    during the live window (so scores refresh) and once per ET day otherwise
    (so an idle page never re-hammers the API). See ``cache_bucket``."""
    try:
        df = dl.load_long(workbook_bytes(source), use_api=use_api)
        df.attrs["origin"] = "live Google Sheet" if source == "live" else "local snapshot"
        return df
    except Exception as exc:  # noqa: BLE001 - surface, then degrade gracefully
        df = dl.load_long(SNAPSHOT, use_api=use_api)
        df.attrs["origin"] = f"local snapshot (live failed: {exc})"
        return df


@st.cache_data(ttl=3600, show_spinner=False)
def kickoff_times(source: str) -> list[pd.Timestamp]:
    """Unique fixture kickoff times (UTC, naive). Pulled WITHOUT the live API —
    fixtures are stable — and cached an hour, so deciding whether we're in the
    live window never touches the flaky results feed."""
    sched = dl.load_long(workbook_bytes(source), use_api=False)
    return sorted(pd.Timestamp(t) for t in sched["datetime"].dropna().unique())


def in_live_window(kicks: list[pd.Timestamp], now_et: pd.Timestamp) -> bool:
    """True when ``now_et`` (New York wall-clock, naive) sits between a match
    day's first kickoff and ``LIVE_END_HOUR``:00 ET the next morning — the
    'Live Updates' window. Fixture times are stored in UK local time, so we
    convert them to ET before grouping: a match day is an ET calendar day, even
    though it may span two UK days. Games past midnight keep the window open
    until the cutoff."""
    if not kicks:
        return False
    uk = pd.Series(pd.to_datetime(kicks)).dt.tz_localize(
        DISPLAY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    et = uk.dt.tz_convert(MATCHDAY_TZ).dt.tz_localize(None).dropna()
    for day in et.dt.normalize().unique():
        day = pd.Timestamp(day)
        start = et[et.dt.normalize() == day].min()
        end = day + pd.Timedelta(days=1, hours=LIVE_END_HOUR)
        if start <= now_et <= end:
            return True
    return False


def cache_bucket(live: bool, now: pd.Timestamp) -> str:
    """Cache differentiator for ``get_data``: a fresh 2-minute slot while live
    (forces refetch every ``LIVE_TTL`` s), else a stable per-ET-day key (one
    load per visit until the day rolls or 'Refresh now')."""
    if live:
        return f"live-{int(now.timestamp()) // LIVE_TTL}"
    # Off the live window, refresh on a 5-min slot (not once per day) so
    # predictions entered in the Sheet show up the same matchday without a
    # manual refresh.
    return f"slot-{int(now.timestamp()) // 300}"


def color_map(players: list[str]) -> dict[str, str]:
    return {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(players)}


@st.cache_data(show_spinner=False)
def played_order(df: pd.DataFrame) -> pd.DataFrame:
    """Played matches with a sequential play-order index (1..N) by kickoff.

    Cached: pure function of ``df`` (which only changes when ``get_data``'s
    cache refreshes), but called on several tabs every rerun — recomputing it
    each time was wasted work."""
    played = df[df["played"]].copy()
    order = (played[["match_id", "datetime"]].drop_duplicates()
             .sort_values("datetime").reset_index(drop=True))
    order["play_order"] = range(1, len(order) + 1)
    order["label"] = order["play_order"].astype(str)
    return played.merge(order[["match_id", "play_order", "label"]], on="match_id")


@st.cache_data(show_spinner=False)
def cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Wide-ish long frame: cumulative points + rank per player per play_order.

    Cached for the same reason as ``played_order``: called on three tabs every
    rerun, but fully determined by ``df``."""
    p = played_order(df)
    grid = (p.groupby(["play_order", "player"])["points"].sum().reset_index())
    full = (pd.MultiIndex.from_product(
        [sorted(p["play_order"].unique()), sorted(p["player"].unique())],
        names=["play_order", "player"]).to_frame(index=False))
    grid = full.merge(grid, on=["play_order", "player"], how="left").fillna({"points": 0})
    grid = grid.sort_values(["player", "play_order"])
    grid["cum"] = grid.groupby("player")["points"].cumsum()
    grid["rank"] = grid.groupby("play_order")["cum"].rank(method="min", ascending=False)
    return grid


@st.cache_data(show_spinner=False)
def matchday_index(df: pd.DataFrame) -> pd.DataFrame:
    """Played rows tagged with their matchday number (1..N), where a matchday
    is one New York calendar day (see data_loader.et_match_day) carrying at
    least one played match."""
    p = df[df["played"]].copy()
    p["mdate"] = p["match_day"]
    lut = {d: i + 1 for i, d in enumerate(sorted(p["mdate"].unique()))}
    p["md"] = p["mdate"].map(lut)
    return p


@st.cache_data(show_spinner=False)
def standings_table(df: pd.DataFrame, players: list[str]):
    """Build the Table-tab standings, best first.

    Columns: Player, Pts, Correct Scores (exact 3-pointers), Correct Results
    (outcome-only 1-pointers), then one points column per matchday (MD1..MDn).
    Returns (table, md_cols, md_help) where md_help maps each MDk to its date.
    """
    p = matchday_index(df)
    pts = p.groupby("player")["points"].sum()
    exact = df[df["exact_hit"]].groupby("player").size()
    outcome = df[df["outcome_only"]].groupby("player").size()

    gained = p.groupby(["player", "md"])["points"].sum().unstack(fill_value=0)
    # Reverse-chronological: most recent matchday first (MDn … MD1).
    md_nums = sorted(gained.columns, reverse=True)
    gained = gained[md_nums]
    gained.columns = [f"MD{n}" for n in md_nums]
    dates = sorted(p["mdate"].unique())
    md_help = {f"MD{i + 1}": pd.Timestamp(d).strftime("%a %-d %b")
               for i, d in enumerate(dates)}

    base = pd.DataFrame({
        "Player": players,
        "Pts": [int(pts.get(pl, 0)) for pl in players],
        "_exact": [int(exact.get(pl, 0)) for pl in players],
        "Correct Results": [int(outcome.get(pl, 0)) for pl in players],
    }).set_index("Player")
    table = (base.join(gained).fillna(0).astype(int).reset_index()
             .sort_values(["Pts", "_exact", "Player"],
                          ascending=[False, False, True]).reset_index(drop=True))
    table.index = table.index + 1
    table.index.name = "#"
    # Lead Correct Scores with the points those exacts earned: 3×X (X correct).
    table.insert(2, "Correct Scores",
                 table["_exact"].map(lambda x: f"{x * 3} ({x} correct)"))
    table = table.drop(columns="_exact")
    return table, list(gained.columns), md_help


@st.cache_data(show_spinner=False)
def time_at_top(df: pd.DataFrame, players: list[str]) -> pd.Series:
    """Share (%) of played matches each player has spent in outright 1st place,
    measured across the cumulative-points race."""
    grid = cumulative(df)
    steps = grid["play_order"].nunique()
    share = grid[grid["rank"] == 1].groupby("player").size() / steps * 100
    return share.reindex(players).fillna(0.0)


@st.cache_data(show_spinner=False)
def time_at_bottom(df: pd.DataFrame, players: list[str]) -> pd.Series:
    """Share (%) of played matches each player has spent in last place (the
    worst rank at that point — ties at the bottom all count)."""
    grid = cumulative(df).copy()
    steps = grid["play_order"].nunique()
    grid["worst"] = grid.groupby("play_order")["rank"].transform("max")
    bottom = grid[grid["rank"] == grid["worst"]]
    share = bottom.groupby("player").size() / steps * 100
    return share.reindex(players).fillna(0.0)


# --------------------------------------------------------------------------- #
# Title-race chart (race + bump) — ONE ECharts figure, shared legend
#
# Two stacked panels (cumulative points on top, rank below) that *run* on open:
# a hidden timeline replays the season match-by-match across both panels at once
# (no visible scrubber), so the lines race in. One legend up top controls both.
# Minimalist — slim lines, no gridlines, no markers, only the leader labelled
# (with a 🏆). Falls back to Plotly if ECharts is missing.
# --------------------------------------------------------------------------- #
def _player_colors(players: list[str]) -> list[str]:
    return [CMAP[p] for p in players]


def echarts_combined_option(df: pd.DataFrame, leader: str) -> dict:
    grid_df = cumulative(df)
    players = sorted(grid_df["player"].unique())
    colors = _player_colors(players)
    orders = sorted(int(o) for o in grid_df["play_order"].unique())
    xmax = max(orders)
    # Fix the points axis once (ceiling + coarse step) so it doesn't rescale on
    # every frame — that constant relabelling is what makes the race stutter.
    cmax = int(grid_df["cum"].max())
    ystep = 10 if cmax > 40 else 5
    ymax = ((cmax // ystep) + 1) * ystep
    data = {(p, y): grid_df[grid_df.player == p].sort_values("play_order")[
        ["play_order", y]].astype(float).values.tolist()
        for p in players for y in ("cum", "rank")}

    def base(p, i, axis, smooth):
        return {
            "name": p, "type": "line", "smooth": smooth, "showSymbol": False,
            "xAxisIndex": axis, "yAxisIndex": axis,
            "lineStyle": {"width": 3 if p == leader else 1.6, "color": colors[i]},
            "itemStyle": {"color": colors[i]},
            "emphasis": {"focus": "series"},
            "endLabel": {"show": False},  # set per-frame to whoever's leading
        }

    base_series = ([base(p, i, 0, True) for i, p in enumerate(players)]
                   + [base(p, i, 1, False) for i, p in enumerate(players)])

    def seg(p, y, f):
        return [pt for pt in data[(p, y)] if pt[0] <= f]

    steps = []
    for f in orders:
        cum_now = {p: (seg(p, "cum", f)[-1][1] if seg(p, "cum", f) else -1)
                   for p in players}
        lead_f = max(players, key=lambda p: cum_now[p])
        # Label only the current leader's line, so the name hops as the lead
        # changes hands. The 🏆 only joins on the final frame (the winner).
        suffix = "  🏆" if f == orders[-1] else ""
        labels = {p: {"show": p == lead_f, "formatter": f"{p}{suffix}",
                      "color": colors[i], "fontWeight": "bold"}
                  for i, p in enumerate(players)}
        race = [{"data": seg(p, "cum", f), "endLabel": labels[p]} for p in players]
        bump = [{"data": seg(p, "rank", f), "endLabel": labels[p]} for p in players]
        steps.append({"series": race + bump})

    axis_common = {"type": "value", "min": 1, "max": xmax, "minInterval": 1,
                   "splitLine": {"show": False}}
    return {
        "baseOption": {
            "color": colors,
            "legend": {"data": players, "top": 6, "type": "scroll", "icon": "roundRect"},
            "tooltip": {"trigger": "axis"},
            "title": [
                {"text": "Position by matchday", "top": 396, "left": 56,
                 "textStyle": {"fontSize": 14, "color": C["navy"]}},
            ],
            "grid": [
                {"top": 72, "left": 58, "right": 112, "height": 268},
                {"top": 426, "left": 58, "right": 112, "height": 210},
            ],
            "xAxis": [
                {**axis_common, "gridIndex": 0},
                {**axis_common, "gridIndex": 1, "name": "Match played",
                 "nameLocation": "middle", "nameGap": 28},
            ],
            "yAxis": [
                {"type": "value", "gridIndex": 0, "min": 0, "max": ymax,
                 "interval": ystep, "name": "Points", "splitLine": {"show": False}},
                {"type": "value", "gridIndex": 1, "inverse": True, "min": 1,
                 "max": len(players), "minInterval": 1, "name": "Position",
                 "splitLine": {"show": False}},
            ],
            "series": base_series,
            "timeline": {"show": False, "autoPlay": True, "loop": False,
                         "playInterval": 300, "axisType": "category",
                         "data": [str(f) for f in orders]},
            "animationDuration": 500,
        },
        "options": steps,
    }


def _plotly_lines(df: pd.DataFrame, y: str, *, reverse: bool, title: str) -> go.Figure:
    """Static Plotly fallback for the race/bump when ECharts is unavailable."""
    grid = cumulative(df)
    fig = go.Figure()
    for p in sorted(grid["player"].unique()):
        g = grid[grid.player == p]
        fig.add_trace(go.Scatter(x=g["play_order"], y=g[y], mode="lines",
                                 line=dict(color=CMAP[p], width=2), name=p))
    fig.update_layout(height=440, xaxis_title="Match played", yaxis_title=title,
                      legend=dict(orientation="h", y=1.12, x=0))
    if reverse:
        fig.update_yaxes(autorange="reversed", dtick=1)
    return fig


# --------------------------------------------------------------------------- #
# Leader / Loser hero cards
# --------------------------------------------------------------------------- #
LL_CSS = """
<style>
.ll-row{display:flex;gap:16px;margin:6px 0 20px;flex-wrap:wrap;}
.ll-card{flex:1;min-width:240px;display:flex;align-items:center;gap:16px;
  border-radius:16px;padding:16px 22px;box-shadow:0 6px 20px rgba(30,58,95,.07);}
.ll-gold{background:linear-gradient(135deg,#FFF7DE,#FDEBB6);border:1px solid #F4C430;}
.ll-silver{background:linear-gradient(135deg,#F6F8FB,#E4E9F0);border:1px solid #C9D2DE;}
.ll-poo{background:linear-gradient(135deg,#F4F1EC,#E7E0D6);border:1px solid #C9BCA8;}
.ll-emoji{font-size:2.6rem;line-height:1;
  font-family:"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif;}
.ll-label{font-size:.78rem;font-weight:800;letter-spacing:.06em;
  text-transform:uppercase;color:#6B7280;}
.ll-name{font-size:1.6rem;font-weight:800;color:#1C1919;line-height:1.1;}
.ll-pts{margin-left:auto;font-size:2.1rem;font-weight:800;color:#1E3A5F;}
.ll-pts span{font-size:.9rem;font-weight:600;color:#6B7280;margin-left:4px;}
</style>
"""


def leader_loser_html(leader_row, runner_row, loser_row) -> str:
    def card(cls, emoji, label, name, pts):
        return (f"<div class='ll-card {cls}'><div class='ll-emoji'>{emoji}</div>"
                f"<div><div class='ll-label'>{label}</div>"
                f"<div class='ll-name'>{name}</div></div>"
                f"<div class='ll-pts'>{pts}<span>pts</span></div></div>")
    return ("<div class='ll-row'>"
            + card("ll-gold", "🥇", "Leader", leader_row["Player"], int(leader_row["Pts"]))
            + card("ll-silver", "🥈", "Runner up", runner_row["Player"], int(runner_row["Pts"]))
            + card("ll-poo", "💩", "poopy poopy poopy", loser_row["Player"], int(loser_row["Pts"]))
            + "</div>")


def _row_highlight(row, me, leader):
    """Tint your row blue and the leader's row gold in the standings table."""
    if row["Player"] == me:
        return ["background-color:#E8EEF7;font-weight:700"] * len(row)
    if row["Player"] == leader:
        return ["background-color:#FFF6E0;font-weight:700"] * len(row)
    return [""] * len(row)


# Standard bookmaker price ladder (odds-to-1 as a decimal → fractional label),
# capped below 100/1 so that infamous price stays Reddy's and Reddy's alone.
_ODDS_LADDER = [
    (2, "2/1"), (2.5, "5/2"), (3, "3/1"), (3.5, "7/2"), (4, "4/1"),
    (4.5, "9/2"), (5, "5/1"), (6, "6/1"), (7, "7/1"), (8, "8/1"),
    (9, "9/1"), (10, "10/1"), (12, "12/1"), (14, "14/1"), (16, "16/1"),
    (20, "20/1"), (25, "25/1"), (33, "33/1"), (40, "40/1"), (50, "50/1"),
    (66, "66/1"), (80, "80/1"),
]


def _snap_odds(value: float) -> str:
    """Nearest standard fractional price to a decimal odds-to-1 figure."""
    return min(_ODDS_LADDER, key=lambda lp: abs(lp[0] - value))[1]


def title_odds(table: pd.DataFrame) -> dict[str, str]:
    """Win-the-whole-thing prices, table sorted best-first.

    Leader: firms up the more daylight they have over 2nd place — 1/2 (>5 clear),
    5/6 (>3), 6/4 (>2), 15/8 (1–2 clear), 2/1 (level at the top).
    Chasers: 2/1 level with the leader, 5/2 one back, then drift out convexly so
    each further point behind costs progressively more. When two-plus share the
    lead, chasers must overhaul more than one rival, so their prices drift out a
    touch further (more co-leaders → longer).
    Reddy: a standing joke — always 100/1.
    """
    players = list(table["Player"])
    pts = [int(x) for x in table["Pts"]]
    lead = pts[0]
    second = pts[1] if len(pts) > 1 else lead
    n_top = pts.count(lead)  # how many share the summit
    odds: dict[str, str] = {}
    for i, pl in enumerate(players):
        if pl == "Reddy":
            odds[pl] = "100/1"
            continue
        if i == 0:
            cushion = lead - second  # points clear of the field
            if cushion > 5:
                odds[pl] = "1/2"
            elif cushion > 3:
                odds[pl] = "5/6"
            elif cushion > 2:
                odds[pl] = "6/4"
            elif cushion >= 1:
                odds[pl] = "15/8"
            else:
                odds[pl] = "2/1"  # level at the summit
        else:
            d = lead - pts[i]  # points behind the leader
            # Anchored at 2/1 (d=0) and 5/2 (d=1); the d·(d−1) term is zero at
            # both anchors and adds simple convexity beyond them.
            price = 2 + 0.5 * d + 0.25 * d * (d - 1)
            # Strictly behind and the lead is shared? Nudge the price out a
            # little for each extra co-leader to climb over (≈15% per rival).
            if d > 0 and n_top >= 2:
                price *= 1 + 0.15 * (n_top - 1)
            odds[pl] = _snap_odds(price)
    return odds


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🛠️ Dev controls")
st.sidebar.info("dev/bug fixing only")
source = st.sidebar.radio("Predictions source", ["live", "snapshot"],
                          format_func=lambda s: {"live": "🔄 Live Google Sheet",
                                                 "snapshot": "💾 Local snapshot"}[s])
use_api = st.sidebar.toggle("⚽ Live results from API", value=True,
                            help="Actual scores from worldcup26.ir; points recompute "
                                 "automatically. Off = use scores typed in the Sheet.")

# Live window (ET match day): first kickoff -> LIVE_END_HOUR:00 ET next morning.
now_et = pd.Timestamp.now(tz=MATCHDAY_TZ).tz_localize(None)
in_window = in_live_window(kickoff_times(source), now_et)
live = in_window and use_api  # only poll the API hard when it's actually feeding

if in_window:
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin:2px 0 4px;
             padding:8px 12px;border-radius:8px;background:rgba(220,38,38,0.12);
             border:1px solid rgba(220,38,38,0.45);font-weight:700;color:#DC2626;">
          <span style="width:9px;height:9px;border-radius:50%;background:#DC2626;
                display:inline-block;animation:lvpulse 1.3s ease-in-out infinite;"></span>
          Live Updates
        </div>
        <style>@keyframes lvpulse{0%,100%{opacity:1;transform:scale(1)}
          50%{opacity:.3;transform:scale(.6)}}</style>
        """,
        unsafe_allow_html=True)
    st.sidebar.caption("⚡ Scores refresh every 2 min")

if st.sidebar.button("Refresh now", width="stretch"):
    workbook_bytes.clear()  # force a fresh Sheet download, not just a re-parse
    get_data.clear()
    kickoff_times.clear()

df = get_data(source, use_api, cache_bucket(live, now_et))

# While live, rerun the page on the cache cadence so scores actually move.
if live:
    st.components.v1.html(
        f"<script>setTimeout(()=>window.parent.location.reload(), {LIVE_TTL * 1000});</script>",
        height=0)
PLAYERS = dl.players(df)
CMAP = color_map(PLAYERS)
ME = None  # no "you" highlight — the leader is the only emphasised row/line
n_played = int(df[df.player == PLAYERS[0]]["played"].sum())


def revealed_matchdays(frame: pd.DataFrame) -> list[tuple[int, pd.Timestamp]]:
    """Matchdays whose 12:00 UK reveal time has passed, as (number, date).

    A matchday becomes 'current' — and selectable — from midday UK on its own
    date; future matchdays are hidden until then. Numbered chronologically
    against the full schedule so the count tracks the real matchday number."""
    days = sorted(pd.Timestamp(d) for d in frame["match_day"].dropna().unique())
    now_uk = pd.Timestamp.now(tz=ZoneInfo("Europe/London"))
    return [(i + 1, d) for i, d in enumerate(days)
            if d.tz_localize("Europe/London") + pd.Timedelta(hours=12) <= now_uk]


st.sidebar.caption(f"Predictions: {df.attrs.get('origin', '?')}")
st.sidebar.caption(f"Results: {df.attrs.get('results_origin', '?')}")
st.sidebar.caption(f"{n_played} matches played · {df['match_id'].nunique()} scheduled")

st.title("⚽ World Cup 2026 Predictions there Boyz")

# Player tab temporarily disabled (WIP) — re-add ":material/person: Player" to
# the list and unpack `tab_player` to restore it. See the `if False:` block below.
tab_table, tab_mdc = st.tabs(
    [":material/leaderboard: Table", ":material/stadium: Matchday Center"])

# --------------------------------------------------------------------------- #
# Live Table
# --------------------------------------------------------------------------- #
with tab_table:
    table, md_cols, md_help = standings_table(df, PLAYERS)
    LEADER = table.iloc[0]["Player"]
    current = revealed_matchdays(df)
    md_now = current[-1][0] if current else 0
    total = df["match_id"].nunique()
    pct = round(100 * n_played / total) if total else 0

    # ---- "As it stands" banner ------------------------------------------ #
    st.markdown(f"### :material/sports_soccer: As it Stands — Matchday {md_now}")
    st.markdown(f"<span style='font-size:1.05rem;color:#6B7280'>"
                f"{n_played} games played · {pct}% complete</span>",
                unsafe_allow_html=True)

    # ---- Leader / Loser hero cards -------------------------------------- #
    st.markdown(LL_CSS, unsafe_allow_html=True)
    st.markdown(leader_loser_html(table.iloc[0], table.iloc[1], table.iloc[-1]),
                unsafe_allow_html=True)

    st.subheader(":material/emoji_events: Table")
    col_cfg = {"Pts": st.column_config.ProgressColumn(
        "Pts", min_value=0, max_value=int(table["Pts"].max() or 1), format="%d")}
    for c in md_cols:
        col_cfg[c] = st.column_config.NumberColumn(c, help=md_help.get(c), width="small")
    # Fade the per-matchday columns so the headline stats (Pts, Correct …) lead.
    styled = (table.style.apply(_row_highlight, axis=1, me=ME, leader=LEADER)
              .set_properties(subset=md_cols, color="#9CA3AF"))
    st.dataframe(styled, width="stretch", column_config=col_cfg)

    # ---- Title race: cumulative race + rank bump, one shared-legend figure #
    st.divider()
    st.subheader(":material/show_chart: The title race")

    # ---- Time share cards: most time in front, and most time at the bottom #
    def _share_cards(series, anchor_player, anchor_label, medals):
        anchor = float(series.get(anchor_player, 0.0))
        top3 = series.sort_values(ascending=False).head(3)
        for col, medal, (p, share) in zip(st.columns(3), medals, top3.items()):
            d = float(share) - anchor
            col.metric(f"{medal} {p}", f"{share:.0f}%",
                       None if abs(d) < 0.5 else f"{d:+.0f}% {anchor_label}")

    with st.expander("Time at the top & bottom", expanded=False):
        st.markdown("**:material/trending_up: Most time in front**")
        _share_cards(time_at_top(df, PLAYERS), LEADER, "vs leader",
                     ["🥇", "🥈", "🥉"])

        st.markdown("**:material/trending_down: Most time at the bottom**")
        _share_cards(time_at_bottom(df, PLAYERS), table.iloc[-1]["Player"],
                     "vs loser", ["💩💩💩", "💩💩", "💩"])

    # NB: deliberately *not* calling streamlit_extras.style_metric_cards here —
    # it injects global CSS (navy border + shadow) onto every st.metric in the
    # app. We prefer Streamlit's plain default cards (the look on the cloud
    # deploy, where streamlit_extras isn't installed).

    # ---- Combined race + bump (shared legend, animates on open) --------- #
    if _HAS_ECHARTS:
        st_echarts(echarts_combined_option(df, LEADER), height="700px", key="race")
    else:
        st.plotly_chart(_plotly_lines(df, "cum", reverse=False, title="Points"),
                        width="stretch")
        st.plotly_chart(_plotly_lines(df, "rank", reverse=True, title="Position"),
                        width="stretch")

    # ---- Heatmap: points by player (merged in under the race chart) ------ #
    st.divider()
    st.subheader(":material/grid_view: Points by player")
    by = st.radio("Group by", ["By match", "By matchday"], horizontal=True,
                  label_visibility="collapsed")

    # Same grey → orange → navy ramp in both views, so a cell's colour means the
    # same thing whether you're looking match-by-match or matchday-by-matchday.
    colorscale = [[0, "#F3F4F6"], [0.33, "#FED7AA"], [1.0, C["navy"]]]
    if by == "By match":
        p = played_order(df)
        p["lbl"] = (p["play_order"].astype(str) + ". "
                    + p["home"].str[:3].str.upper() + "–" + p["away"].str[:3].str.upper())
        pivot = (p.pivot_table(index="player", columns="play_order",
                               values="points", aggfunc="sum").reindex(PLAYERS))
        labels = (p[["play_order", "lbl"]].drop_duplicates()
                  .sort_values("play_order")["lbl"].tolist())
        zmax = 3  # one match = at most an exact 3
    else:
        p = matchday_index(df)
        pivot = (p.pivot_table(index="player", columns="md",
                               values="points", aggfunc="sum").reindex(PLAYERS))
        pivot = pivot[sorted(pivot.columns)]
        labels = [f"MD{n}" for n in sorted(pivot.columns)]
        zmax = int(max(3, pivot.max().max()))  # a matchday stacks several games

    heat = go.Figure(go.Heatmap(
        z=pivot.values, x=labels, y=pivot.index,
        colorscale=colorscale, zmin=0, zmax=zmax, xgap=2, ygap=2,
        showscale=False,
        hovertemplate="%{y}<br>%{x}<br>%{z} pts<extra></extra>"))
    heat.update_layout(height=380, xaxis=dict(tickangle=-60, showgrid=False),
                       yaxis=dict(showgrid=False))
    st.plotly_chart(heat, width="stretch")

# --------------------------------------------------------------------------- #
# Matchday Center
# --------------------------------------------------------------------------- #
with tab_mdc:
    st.subheader(":material/stadium: Matchday Center")
    # Only matchdays revealed so far (from 12:00 UK on their own date) — no
    # future matchdays in the picker. Default to the latest revealed one.
    rounds = revealed_matchdays(df)
    if not rounds:
        st.info("No matchday is live yet — check back from midday on the first matchday.")
    else:
        rlabels = {idx: f"MD{idx} · {d:%a %-d %b}" for idx, d in rounds}
        sel = st.selectbox("Matchday", [idx for idx, _ in rounds],
                           index=len(rounds) - 1, format_func=lambda i: rlabels[i])
        sel_date = dict(rounds)[sel]

        day = df[df["match_day"] == sel_date]
        mids = day.sort_values("datetime")["match_id"].unique()
        n_total = len(mids)
        n_played = int(day.groupby("match_id")["played"].first().sum())
        completed = n_played == n_total
        has_pred = bool(day["has_prediction"].any())


        # ============================================================ #
        # ✏️ COPY — Matchday status line
        #   status_txt: the done / in-progress / upcoming pill text.
        #   st.caption: the one-line description under the picker.
        #   Available vars: n_total, n_played, rlabels[sel] (e.g. "MD10 · Sat 20 Jun").
        # ============================================================ #
        matches_word = "match" if n_total == 1 else "matches"
        first_ko = pd.to_datetime(day["datetime"]).min()
        if completed:
            lead = f"{n_total} {matches_word}" if n_total == 1 else f"all {n_total} {matches_word}"
            status_txt = f"✅ Completed · {lead} played"
        elif n_played == 0:
            status_txt = f"⏰ Kick Off {first_ko:%H:%M}"
        else:
            status_txt = f"🔴 In progress · {n_played} of {n_total} {matches_word} played"
        st.caption(f"Fixtures, **{rlabels[sel]}** &nbsp;·&nbsp; {status_txt}")
        # --- end COPY: status line --- #

        # ---- Matchday recap (deterministic, screenshot-ready) ----------- #
        # Recap is results-driven, so only build it once this matchday has at
        # least one played game; upcoming matchdays get a lighter note instead.
        recap = extras.build_recap(df, PLAYERS, round_index=sel, me=ME) if n_played else None
        if not recap:
            note = ("Predictions are locked in — fixtures below."
                    if has_pred else
                    "Predictions aren't uploaded yet — fixtures below, scores and "
                    "recap will fill in once picks and results land.")
            st.info(f"⏰ **{rlabels[sel]}** is still to come. {note}")
        if recap:
            try:
                from streamlit_extras.stylable_container import stylable_container
                card = stylable_container(key="mdc_recap", css_styles="""
                    {
                        border: 1px solid #E6EAF0;
                        border-left: 6px solid #1E3A5F;
                        border-radius: 14px;
                        padding: 16px 22px;
                        box-shadow: 0 6px 18px rgba(30,58,95,0.07);
                    }""")
            except Exception:
                card = st.container(border=True)
            with card:
                # ==================================================== #
                # ✏️ COPY — Recap card header
                #   tag: the "X of Y games played" subtitle.
                #   The "{recap['title']} so far" headline (📣 line).
                #   recap['lines'] themselves are generated in extras.py
                #   build_recap() — edit that function to reword them.
                # ==================================================== #
                # Count the played matches (n_played), not just games that
                # scored, so this lines up with the fixtures shown below.
                if completed:
                    tag = f"{n_total} games"
                else:
                    tag = f"{n_played} of {n_total} games played"
                st.markdown(
                    f"#### 📣 {recap['title']} so far &nbsp;"
                    f"<span style='color:#6B7280;font-weight:400;font-size:0.9rem'>"
                    f"{recap['date_label']} · {tag}</span>",
                    unsafe_allow_html=True)
                st.markdown("  \n".join(recap["lines"]))
                # --- end COPY: recap card --- #

        # ---- Latest odds: each player's price to win it outright ---------- #
        odds = title_odds(standings_table(df, PLAYERS)[0])
        st.markdown("**:material/casino: Latest odds** &nbsp;"
                    "<span style='color:#6B7280;font-weight:400;font-size:0.85rem'>"
                    "source: Patrick Power (feed delayed 15 mins)</span>"
                    "<span style='display:block;line-height:1;margin-top:-4px;"
                    "color:#9CA3AF;font-weight:400;font-size:0.7rem'>"
                    "Please Gamble Responsibly</span>",
                    unsafe_allow_html=True)
        ocols = st.columns(len(odds))
        for col, (pl, price) in zip(ocols, odds.items()):
            col.metric(pl, price)

        st.divider()
        # ============================================================ #
        # ✏️ COPY — Fixtures section heading + caption
        #   The "#### Fixtures & predictions" heading.
        #   st.caption: the one-liner explaining what each box shows.
        #   (Per-box wording lives in cards.py match_card_html — see the
        #    "✏️ COPY" markers there.)
        # ============================================================ #
        st.markdown("#### :material/sports_soccer: Fixtures & predictions")
        st.caption(f"{n_total} match{'es' if n_total != 1 else ''} this matchday")
        # --- end COPY: fixtures heading --- #
        st.markdown(cardui.CSS, unsafe_allow_html=True)
        for mid in mids:
            st.markdown(cardui.match_card_html(df[df.match_id == int(mid)]),
                        unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Player profile  — TEMPORARILY DISABLED (WIP). To re-enable: restore the
# `tab_player` tab above and change `if False:` back to `with tab_player:`.
# --------------------------------------------------------------------------- #
if False:  # noqa: SIM223 - parked tab, intentionally not rendered for now
    st.subheader(":material/person: Player profile")
    who = st.selectbox("Player", PLAYERS,
                       index=PLAYERS.index("Kian") if "Kian" in PLAYERS else 0)
    pl = df[df.player == who]
    plp = pl[pl.played]
    grid = cumulative(df)
    rank_now = int(grid[(grid.player == who) & (grid.play_order == grid.play_order.max())]["rank"].iloc[0])

    played_all = df[df.played]
    stats = pd.DataFrame({
        "points": played_all.groupby("player")["points"].sum(),
        "exact": played_all.groupby("player")["exact_hit"].sum(),
        "outcome": played_all.groupby("player")["outcome_only"].sum(),
        "miss": played_all.groupby("player")["miss"].sum(),
    }).reindex(PLAYERS).fillna(0)
    st.markdown(cardui.CSS, unsafe_allow_html=True)
    st.markdown(cardui.stat_cards_html(cardui.player_stat_items(stats, who, rank_now)),
                unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Points by stage**")
        by_stage = (plp.groupby("stage")["points"].sum().reset_index())
        sfig = px.bar(by_stage, x="stage", y="points", text="points")
        sfig.update_traces(marker_color=CMAP[who], textposition="outside")
        sfig.update_layout(height=300, xaxis_title="", yaxis_title="Points")
        st.plotly_chart(sfig, width="stretch")
    with right:
        st.markdown("**Predictor tendency**")
        pred = plp[plp.has_prediction]
        avg_goals = (pred["pred_home"] + pred["pred_away"]).mean()
        draws = (pred["pred_home"] == pred["pred_away"]).mean() * 100
        st.metric("Avg goals predicted / game", f"{avg_goals:.2f}")
        st.metric("Draws predicted", f"{draws:.0f}%")
        hit_rate = plp["exact_hit"].mean() * 100
        st.metric("Exact-score hit rate", f"{hit_rate:.0f}%")

    st.markdown("**Every prediction so far**")
    show = plp[["home", "actual_score", "away", "pred_score", "points", "stage_raw"]].copy()
    show = show.rename(columns={"home": "Home", "away": "Away",
                                "actual_score": "Result", "pred_score": "Predicted",
                                "points": "Pts", "stage_raw": "Stage"})
    st.dataframe(show.sort_values("Pts", ascending=False),
                 width="stretch", hide_index=True)
