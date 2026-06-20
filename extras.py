"""Presentation helpers for the WC 2026 Pool dashboard.

Three self-contained features, kept out of app.py so the page script stays
readable:

  * setup_plotly_theme()   - one house style for every Plotly figure
  * build_recap()          - a generated "Matchday N" recap (pure pandas)
  * render_share_card()    - a PNG standings card for the group chat (Pillow)
  * aggrid_leaderboard()   - richer standings table, with a graceful fallback

Nothing here imports Streamlit at module load except the small UI wrappers,
so the data functions can be unit-tested headless.
"""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd

# --------------------------------------------------------------------------- #
# Plotly house style
# --------------------------------------------------------------------------- #
def setup_plotly_theme(colors: dict, palette: list[str]) -> None:
    """Register and activate a single template so every chart shares one look
    (white surface, soft gridlines, consistent font and player colours)."""
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates["pool"] = go.layout.Template(layout=dict(
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                  color=colors.get("ink", "#1C1919"), size=13),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=palette,
        title=dict(font=dict(size=18, color=colors.get("navy", "#1E3A5F"))),
        xaxis=dict(showgrid=False, zeroline=False, ticks="outside",
                   ticklen=4, tickcolor="#D7DCE3"),
        yaxis=dict(gridcolor="#ECEFF3", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=48, r=24, t=48, b=44),
    ))
    pio.templates.default = "pool"


# --------------------------------------------------------------------------- #
# Matchday recap
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _names_by_count(series: pd.Series, label: str, hattrick: bool = False) -> str:
    """Group players who share a count onto one phrase, highest count first.

    e.g. {Daithi:1, Garret:1, Noonan:2} -> "**Noonan** 2 Spot on ·
    **Daithi, Garret** 1 Spot on". With hattrick=True, a count of 3+ becomes
    "**Names - Hatrick Hero**" instead of the "N <label>" form.
    """
    buckets: dict[int, list[str]] = {}
    for name, cnt in series.sort_values(ascending=False).items():
        buckets.setdefault(int(cnt), []).append(str(name))
    out = []
    for cnt in sorted(buckets, reverse=True):
        names = ", ".join(buckets[cnt])
        if hattrick and cnt >= 3:
            out.append(f"**{names} - Hatrick Hero**")
        elif label:
            out.append(f"**{names}** {cnt} {label}")
        else:
            out.append(f"**{names}** {cnt}")
    return " · ".join(out)


def _played(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df["played"]].copy()
    # A "matchday" is a New York 24h window (see data_loader.et_match_day), so a
    # match day can span two UK dates. Fall back to UK date only if the column
    # is somehow absent (older cached frame).
    p["date"] = (p["match_day"] if "match_day" in p.columns
                 else pd.to_datetime(p["datetime"]).dt.normalize())
    return p


def list_rounds(df: pd.DataFrame) -> list[tuple[int, pd.Timestamp]]:
    """Each distinct calendar date with a played match is one 'matchday'."""
    dates = sorted(_played(df)["date"].unique())
    return [(i + 1, pd.Timestamp(d)) for i, d in enumerate(dates)]


def _standings_after(df: pd.DataFrame, players: list[str], date) -> tuple[pd.Series, pd.Series]:
    upto = _played(df)
    upto = upto[upto["date"] <= date]
    pts = (upto.groupby("player")["points"].sum()
           .reindex(players).fillna(0).astype(int))
    rank = pts.rank(method="min", ascending=False).astype(int)
    return pts, rank


def standings_rows(df: pd.DataFrame, players: list[str]) -> list[tuple[int, str, int]]:
    """Final standings as (rank, player, points), best first."""
    rounds = list_rounds(df)
    if not rounds:
        return []
    pts, rank = _standings_after(df, players, rounds[-1][1])
    rows = sorted(((int(rank[p]), p, int(pts[p])) for p in players),
                  key=lambda r: (r[0], -r[2], r[1]))
    return rows


def build_recap(df: pd.DataFrame, players: list[str],
                round_index: int | None = None, me: str | None = None) -> dict | None:
    """Generate a punchy recap for one matchday (default: the latest).

    Returns a dict the card UI and the share image both read, or None when no
    match has been played yet. All logic is deterministic - no model calls."""
    rounds = list_rounds(df)
    if not rounds:
        return None
    if round_index is None:
        round_index = rounds[-1][0]
    round_index = max(1, min(round_index, len(rounds)))
    idx, date = rounds[round_index - 1]
    prev_date = rounds[round_index - 2][1] if round_index >= 2 else None

    day = _played(df)
    today = day[day["date"] == date]
    gained = (today.groupby("player")["points"].sum()
              .reindex(players).fillna(0).astype(int))
    exacts = today[today["exact_hit"]].groupby("player").size() if "exact_hit" in today else pd.Series(dtype=int)
    outcomes = (today[today["outcome_only"]].groupby("player").size()
                if "outcome_only" in today else pd.Series(dtype=int))

    pts_after, rank_after = _standings_after(df, players, date)
    if prev_date is not None:
        prev_day = day[day["date"] == prev_date]
        gained_prev = (prev_day.groupby("player")["points"].sum()
                       .reindex(players).fillna(0).astype(int))
        _, rank_before = _standings_after(df, players, prev_date)
        move = (rank_before - rank_after)            # +ve = climbed
    else:
        gained_prev = pd.Series(0, index=players)
        move = pd.Series(0, index=players)

    n_games = today["match_id"].nunique()
    leader = sorted(players, key=lambda p: (rank_after[p], -pts_after[p], p))[0]
    runner = sorted(players, key=lambda p: (rank_after[p], -pts_after[p], p))[1] if len(players) > 1 else leader
    gap = int(pts_after[leader] - pts_after[runner])

    lines: list[str] = []

    # 1) Title picture — always the first line.
    if gap == 0:
        lines.append(f"🏆 **{leader}** & **{runner}** tied at the top on {int(pts_after[leader])}.")
    else:
        lines.append(f"🏆 **{leader}** leads by {gap} over **{runner}**.")

    # 2) Headline: biggest points haul of the day (handle shared tops).
    top_gain = int(gained.max())
    heroes = sorted([p for p in gained.index if int(gained[p]) == top_gain],
                    key=lambda p: rank_after[p])
    if top_gain > 0:
        if len(heroes) == 1:
            hero = heroes[0]
            r = int(rank_after[hero])
            verb = (f"leaps to {_ordinal(r)}" if move.get(hero, 0) > 0
                    else (f"holds {_ordinal(r)}" if r <= 3 else f"climbs to {_ordinal(r)}"
                          if move.get(hero, 0) > 0 else f"sits {_ordinal(r)}"))
            lines.append(f"🏅 Sham of the Match: **{hero} +{top_gain}** — {verb}.")
        else:
            names = ", ".join(heroes)
            lines.append(f"🏅 Sham of the Match: **{names}** — +{top_gain} each.")

    # 3) Biggest rank climb (skip anyone already named Sham of the Match).
    if move.max() > 0:
        climber = move.sort_values(ascending=False).index[0]
        if climber not in heroes:
            lines.append(f"📈 **{climber}** up {int(move[climber])} to {_ordinal(int(rank_after[climber]))}.")

    # A game still in play isn't a verdict: only "shame" lines (nobody nailed it,
    # stone-useless) fire once at least one game today has actually concluded
    # (FT, not live). With just a live first game up, scores aren't final yet.
    if "live" in today.columns and len(today):
        n_concluded = int(today[~today["live"].astype(bool)]["match_id"].nunique())
    else:
        n_concluded = n_games

    # 3) Sharpshooters (exact scores today) — names grouped by count, one line.
    #    The "Innocent Boys" jibe only lands when NOBODY scored at all (no exacts
    #    AND no correct outcomes); if some got the result right it's not 8 of them.
    if len(exacts):
        lines.append("🎯 " + _names_by_count(exacts, "Spot on", hattrick=True))
    elif n_concluded >= 1 and len(outcomes) == 0:
        lines.append(f"🙈 {len(players)} Innocent Boys tell truth")

    # 4) Correct outcomes today (right result, not exact) — names grouped.
    if len(outcomes):
        lines.append("🥈 Correct Outcome — " + _names_by_count(outcomes, ""))

    # 5) Blank watch (scored nothing today) — last line; suppressed until a game
    #    has concluded so we don't pillory anyone mid-first-game. Lists everyone.
    blanks = [p for p in players if gained[p] == 0]
    if blanks and n_concluded >= 1:
        blanks = sorted(blanks, key=lambda p: rank_after[p])
        names = ", ".join(blanks)
        again = (prev_date is not None
                 and all(gained_prev.get(p, 1) == 0 for p in blanks))
        lines.append(f"🪦 **{names} Stone Useless**{' again' if again else ''}")

    return {
        "round": idx,
        "title": f"Matchday {idx}",
        "date_label": pd.Timestamp(date).strftime("%a %-d %b"),
        "n_games": int(n_games),
        "leader": leader,
        "leader_pts": int(pts_after[leader]),
        "runner": runner,
        "gap": gap,
        "lines": lines[:6] if len(lines) > 6 else lines,
        "me": me,
        "me_rank": int(rank_after[me]) if me in players else None,
        "me_pts": int(pts_after[me]) if me in players else None,
    }


# --------------------------------------------------------------------------- #
# Shareable standings PNG (Pillow - no headless-Chrome dependency)
# --------------------------------------------------------------------------- #
_FONT_BOLD = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_REG = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    for path in (_FONT_BOLD if bold else _FONT_REG):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _center(draw, cx, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((cx - (bbox[2] - bbox[0]) / 2, y), text, font=font, fill=fill)


def render_share_card(rows: list[tuple[int, str, int]], me: str | None,
                      subtitle: str) -> bytes:
    """Render a square standings card (top-3 podium + your rank) to PNG bytes.

    `rows` is (rank, player, points) sorted best-first. Pure Pillow so it runs
    anywhere the app does, no extra system binaries."""
    from PIL import Image, ImageDraw

    NAVY, INK, MUTE = "#1E3A5F", "#1C1919", "#6B7280"
    GOLD, SILVER, BRONZE = "#F4C430", "#C9CED6", "#D08B45"
    TINT = "#EEF2F8"
    W = H = 1080
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # Header band
    d.rectangle([0, 0, W, 168], fill=NAVY)
    _center(d, W / 2, 40, "WC 2026 POOL", _font(58, bold=True), "white")
    _center(d, W / 2, 112, subtitle, _font(28), "#C7D2E3")

    # Podium (centre = 1st, left = 2nd, right = 3rd)
    base_y = 740
    spec = [(1, W / 2, 360, GOLD), (2, W / 2 - 260, 270, SILVER), (3, W / 2 + 260, 210, BRONZE)]
    by_rank = {r: (p, pts) for r, p, pts in rows}
    bar_w = 220
    for rank, cx, h, col in spec:
        if rank not in by_rank:
            continue
        name, pts = by_rank[rank]
        top = base_y - h
        d.rounded_rectangle([cx - bar_w / 2, top, cx + bar_w / 2, base_y],
                            radius=18, fill=col)
        # rank numeral inside the bar
        _center(d, cx, top + 24, str(rank), _font(72, bold=True),
                "white" if rank != 1 else NAVY)
        # name + points above the bar
        _center(d, cx, top - 86, name, _font(40, bold=True), INK)
        _center(d, cx, top - 40, f"{pts} pts", _font(30), MUTE)
    d.line([90, base_y, W - 90, base_y], fill="#D7DCE3", width=3)

    # "Your rank" strip
    me_row = next((r for r in rows if r[1] == me), None)
    if me_row:
        rk, _, pts = me_row
        lead_pts = rows[0][2]
        off = lead_pts - pts
        strip_y = 812
        d.rounded_rectangle([90, strip_y, W - 90, strip_y + 150], radius=20, fill=TINT)
        d.rectangle([90, strip_y, 102, strip_y + 150], fill=NAVY)
        d.text((140, strip_y + 28), "YOUR RANK", font=_font(24, bold=True), fill=MUTE)
        if rk == 1:
            tail = "top of the pile 🏆" if False else "— top of the table"
        else:
            tail = f"— {off} off the lead" if off > 0 else "— level with the lead"
        d.text((140, strip_y + 64), f"#{rk}  {me}", font=_font(46, bold=True), fill=NAVY)
        _center(d, W - 250, strip_y + 74, f"{pts} pts {tail}", _font(28), INK)

    _center(d, W / 2, H - 56,
            f"generated {datetime.now():%-d %b %Y} · worldcup26.ir live feed",
            _font(22), MUTE)

    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# --------------------------------------------------------------------------- #
# AgGrid leaderboard (with graceful fallback)
# --------------------------------------------------------------------------- #
def aggrid_leaderboard(table: pd.DataFrame, me: str | None) -> bool:
    """Render the standings via st_aggrid with a highlighted leader + 'you' row.

    Returns True if AgGrid rendered, False if it was unavailable (caller then
    falls back to st.dataframe)."""
    try:
        import streamlit as st
        from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    except Exception:
        return False

    grid_df = table.reset_index()  # '#' (rank) becomes a real column
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(sortable=True, filter=False, resizable=False,
                                suppressMenu=True)
    gb.configure_column("#", header_name="#", width=58, pinned="left")
    gb.configure_column("Player", width=150, pinned="left")
    gb.configure_column("Move", width=92)
    gb.configure_column("Pts", width=92, type=["numericColumn"])
    gb.configure_column("Exact 3pts", header_name="🎯 Exact", width=110)
    gb.configure_column("Form (last 5)", header_name="🔥 Form", width=104)

    row_style = JsCode("""
    function(params) {
      if (params.data.Player === '%s') {
        return {'backgroundColor': '#E8EEF7', 'fontWeight': '700'};
      }
      if (params.data['#'] === 1) {
        return {'backgroundColor': '#FFF6E0', 'fontWeight': '700'};
      }
      return null;
    }""" % (me or "").replace("'", "\\'"))

    opts = gb.build()
    opts["getRowStyle"] = row_style
    AgGrid(grid_df, gridOptions=opts, allow_unsafe_jscode=True,
           fit_columns_on_grid_load=True, theme="balham",
           height=min(80 + 36 * len(grid_df), 420))
    return True
