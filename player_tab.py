"""Player profile tab for the WC 2026 Pool dashboard.

One deep-dive per league member, hybridised from the formats the group
actually consumes: a FIFA/FUT-style hero card (screenshot bait), betting-site
form chips, an FM-style attribute radar, FBref percentile bars, tendency
stats, and the full prediction log.

All stat computation is pure pandas on the tidy long frame from
data_loader.load_long(); Streamlit only enters in render(). Blank
predictions (no-shows) are first-class throughout — the Sheet scores a blank
as 0-0, so a no-show can fluke points on draws, and the tab says so.
"""
from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cards import flag

try:
    from streamlit_echarts import st_echarts
    _HAS_ECHARTS = True
except Exception:  # pragma: no cover - fall back to Plotly
    _HAS_ECHARTS = False

# Attribute keys in card/radar order (FUT bottom row reads ACC RES BLD / CON FRM REL).
ATTRS = ["ACC", "RES", "BLD", "CON", "FRM", "REL"]
ATTR_HELP = {
    "ACC": "Accuracy — exact-score hit rate",
    "RES": "Results — correct outcome rate (exact or result)",
    "BLD": "Boldness — how often a pick goes against the group's consensus",
    "CON": "Consistency — steadiness of matchday hauls",
    "FRM": "Form — points over the last 10 matches",
    "REL": "Reliability — predictions submitted",
}
LAST_N = 10  # window for form (chips, FRM attribute, scouting lines)


# --------------------------------------------------------------------------- #
# Stats (pure pandas)
# --------------------------------------------------------------------------- #
def _scale(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Min-max a raw stat onto a FIFA-looking range. A flat field (everyone
    identical) collapses to the midpoint rather than dividing by zero."""
    rng = s.max() - s.min()
    if not rng:
        return pd.Series((lo + hi) / 2, index=s.index)
    return lo + (s - s.min()) / rng * (hi - lo)


@st.cache_data(show_spinner=False)
def player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: raw counts, rates, attribute ratings (52-96),
    overall rating (64-93) and league percentiles. Rates that measure *skill*
    (ACC/RES) exclude blanks — the auto 0-0 isn't judgement — while points
    and form include them, matching how the league actually scores."""
    p = df[df["played"]].copy()
    made = p[p["has_prediction"]]

    g = pd.DataFrame(index=sorted(df["player"].unique()))
    g["played"] = p.groupby("player").size()
    g["pts"] = p.groupby("player")["points"].sum()
    g["exacts"] = p[p["exact_hit"]].groupby("player").size()
    g["results"] = p[p["outcome_only"]].groupby("player").size()
    g["blanks"] = p[~p["has_prediction"]].groupby("player").size()
    g["blank_pts"] = p[~p["has_prediction"]].groupby("player")["points"].sum()
    g = g.fillna(0).astype(int)

    g["ppg"] = g["pts"] / g["played"].clip(lower=1)
    g["acc_rate"] = (made[made["exact_hit"]].groupby("player").size()
                     .reindex(g.index).fillna(0)
                     / made.groupby("player").size().reindex(g.index).clip(lower=1))
    g["res_rate"] = (made[made["points"] > 0].groupby("player").size()
                     .reindex(g.index).fillna(0)
                     / made.groupby("player").size().reindex(g.index).clip(lower=1))
    g["rel_rate"] = 1 - g["blanks"] / g["played"].clip(lower=1)

    # Boldness: share of submitted picks whose outcome (H/D/A) disagrees with
    # the group's modal pick for that match — the contrarian index.
    made = made.assign(sig=np.sign(made["pred_home"] - made["pred_away"]))
    modal = made.groupby("match_id")["sig"].agg(lambda s: s.mode().iloc[0])
    made = made.assign(bold=made["sig"] != made["match_id"].map(modal))
    g["bold_rate"] = (made.groupby("player")["bold"].mean()
                      .reindex(g.index).fillna(0))

    # Form: points over the league's last N played matches (same window for
    # everyone, so the comparison is fair even mid-matchday).
    last_ids = (p[["match_id", "datetime"]].drop_duplicates()
                .sort_values("datetime")["match_id"].tail(LAST_N))
    g["last10"] = (p[p["match_id"].isin(last_ids)]
                   .groupby("player")["points"].sum()
                   .reindex(g.index).fillna(0).astype(int))

    # Consistency: low matchday-to-matchday spread = high CON.
    g["md_std"] = (p.groupby(["player", "match_day"])["points"].sum()
                   .groupby("player").std().reindex(g.index).fillna(0))

    g["ACC"] = _scale(g["acc_rate"], 52, 96)
    g["RES"] = _scale(g["res_rate"], 52, 96)
    g["BLD"] = _scale(g["bold_rate"], 52, 96)
    g["CON"] = _scale(-g["md_std"], 52, 96)
    g["FRM"] = _scale(g["last10"], 52, 96)
    g["REL"] = _scale(g["rel_rate"], 52, 96)
    g[ATTRS] = g[ATTRS].round().astype(int)
    g["rating"] = _scale(g["ppg"], 64, 93).round().astype(int)

    g["rank"] = g["pts"].rank(method="min", ascending=False).astype(int)
    # League rank per stat for the bars (1 = best; ties share a rank).
    for col, raw in [("rk_ppg", "ppg"), ("rk_acc", "acc_rate"),
                     ("rk_res", "res_rate"), ("rk_rel", "rel_rate"),
                     ("rk_bold", "bold_rate"), ("rk_frm", "last10")]:
        g[col] = g[raw].rank(ascending=False, method="min").astype(int)
    return g


def rank_before_latest(df: pd.DataFrame) -> pd.Series:
    """Standings rank as of *before* the most recent played matchday, for the
    'moved up/down N' delta. Empty series when only one matchday exists."""
    days = sorted(df.loc[df["played"], "match_day"].dropna().unique())
    if len(days) < 2:
        return pd.Series(dtype=int)
    prev = df[df["played"] & (df["match_day"] < days[-1])]
    pts = prev.groupby("player")["points"].sum()
    return pts.rank(method="min", ascending=False).astype(int)


def form_chips(df: pd.DataFrame, player: str, n: int = LAST_N) -> list[dict]:
    """Last n played matches for one player, oldest first: chip class, label
    and a hover tooltip. Blanks get their own grey chip — a lazy stretch
    should look different from a bad one."""
    rows = (df[(df["player"] == player) & df["played"]]
            .sort_values("datetime").tail(n))
    out = []
    for _, r in rows.iterrows():
        if not r["has_prediction"]:
            cls, lab = "f-b", "–"
        elif r["exact_hit"]:
            cls, lab = "f-x", "3"
        elif r["outcome_only"]:
            cls, lab = "f-o", "1"
        else:
            cls, lab = "f-m", "0"
        pick = str(r["pred_score"]) if r["has_prediction"] else "no pick"
        out.append({"cls": cls, "lab": lab,
                    "tip": (f"{r['home']} {r['actual_score']} {r['away']}"
                            f" · pick {pick} · {int(r['points'])} pt")})
    return out


def team_form(df: pd.DataFrame, player: str, min_games: int = 3):
    """(banker, bogey) — the teams this player reads best and worst, as
    (team, avg pts, games), among teams they've predicted min_games+ times."""
    rows = df[(df["player"] == player) & df["played"] & df["has_prediction"]]
    sides = pd.concat([rows[["home", "points"]].rename(columns={"home": "team"}),
                       rows[["away", "points"]].rename(columns={"away": "team"})])
    agg = sides.groupby("team")["points"].agg(["mean", "count"])
    agg = agg[agg["count"] >= min_games]
    if agg.empty:
        return None, None
    best = agg.sort_values(["mean", "count"], ascending=[False, False]).iloc[0]
    worst = agg.sort_values(["mean", "count"], ascending=[True, False]).iloc[0]
    return ((best.name, best["mean"], int(best["count"])),
            (worst.name, worst["mean"], int(worst["count"])))


def best_pick(df: pd.DataFrame, player: str):
    """The boldest pick that paid off: a scoring pick whose outcome went
    against the group's modal call. Prefer exacts, then the pick that left
    the most rivals on zero, then the most recent. None if they've never
    scored off-consensus."""
    p = df[df["played"] & df["has_prediction"]].copy()
    p["sig"] = np.sign(p["pred_home"] - p["pred_away"])
    modal = p.groupby("match_id")["sig"].agg(lambda s: s.mode().iloc[0])
    mine = p[(p["player"] == player) & (p["points"] > 0)].copy()
    mine = mine[mine["sig"] != mine["match_id"].map(modal)]
    if mine.empty:
        return None
    others = df[(df["player"] != player) & df["played"]]
    zeros = others.groupby("match_id")["points"].agg(lambda s: int((s == 0).sum()))
    mine["zeros"] = mine["match_id"].map(zeros).fillna(0).astype(int)
    return mine.sort_values(["points", "zeros", "datetime"],
                            ascending=[False, False, False]).iloc[0]


def stinkers(df: pd.DataFrame, player: str):
    """(strict, soft) stinker matches. Strict: every other player hit the
    exact 3 and this player alone didn't. Soft: every other player scored
    something and this player alone blanked to zero."""
    p = df[df["played"]]
    per = p[p["player"] != player].groupby("match_id")["points"]
    all3 = per.agg(lambda s: bool((s == 3).all()))
    allpos = per.agg(lambda s: bool((s > 0).all()))
    mine = p[p["player"] == player]
    strict = mine[mine["match_id"].isin(all3[all3].index) & (mine["points"] < 3)]
    soft = mine[mine["match_id"].isin(allpos[allpos].index) & (mine["points"] == 0)]
    return strict, soft


def scoreline_habits(df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Most-predicted scorelines with how often each actually landed."""
    rows = df[(df["player"] == player) & df["played"] & df["has_prediction"]]
    if rows.empty:
        return pd.DataFrame(columns=["Score", "Picked", "Landed"])
    habit = (rows.groupby("pred_score")
             .agg(Picked=("pred_score", "size"), Landed=("exact_hit", "sum"))
             .sort_values("Picked", ascending=False).head(3).reset_index()
             .rename(columns={"pred_score": "Score"}))
    habit["Landed"] = habit["Landed"].astype(int)
    return habit


# Plain-English reads of each attribute for the scouting report, so the
# acronyms on the radar/card get spelled out where the banter lives.
_BEST_TRAIT = {
    "ACC": "Has the whole of England and Ireland bet",
    "RES": "Knows where we're going just not how we're getting there",
    "BLD": "Either a Maverick or a Wet Brain hard to know",
    "CON": "metronomic, steady matchday hauls",
    "FRM": "flying over the last 10 matches",
    "REL": "Fuck all to be at",
}
_WORST_TRAIT = {
    "ACC": "A small fat duck 🦆",
    "RES": "Couldn't pick his nose 👃",
    "BLD": "An auld fence sitter 🚧",
    "CON": "Up and down like a fiddlers elbow 🎻",
    "FRM": "stone useless over the last 10 matches 🪣",
    "REL": "The Sleepy Joe Award 🥇",
}


def scouting_report(df: pd.DataFrame, stats: pd.DataFrame, player: str) -> list[str]:
    """Deterministic banter lines from whichever facts fire for this player —
    same idea as extras.build_recap, no model calls. Capped at 9."""
    s = stats.loc[player]
    rows = (df[(df["player"] == player) & df["played"]]
            .sort_values("datetime").reset_index(drop=True))
    lines: list[str] = []

    habit = scoreline_habits(df, player)
    if len(habit):
        h = habit.iloc[0]
        if int(h["Picked"]) >= 5:
            landed = (f"landed {int(h['Landed'])}" if h["Landed"]
                      else "never landed once")
            lines.append(f"🎯 A **{h['Score']} merchant** — went to that well "
                         f"{int(h['Picked'])} times, {landed}.")
        else:
            # No scoreline used 5+ times — a scattergun, not a merchant.
            lines.append(f"😕 All over the place — most picked "
                         f"**{h['Score']}** {int(h['Picked'])} times.")

    # Standout and weak attributes, echoing the radar in words.
    best = max(ATTRS, key=lambda a: int(s[a]))
    worst = min(ATTRS, key=lambda a: int(s[a]))
    lines.append(f"💪 Best trait **{best} {int(s[best])}** "
                 f"({ATTR_HELP[best].split(' — ')[1]}) — {_BEST_TRAIT[best]}.")
    if worst != best:
        lines.append(f"🚧 Weak spot **{worst} {int(s[worst])}** "
                     f"({ATTR_HELP[worst].split(' — ')[1]}) — {_WORST_TRAIT[worst]}.")

    # Boldest pick that paid — off-consensus and on the money.
    b = best_pick(df, player)
    if b is not None:
        verb = "spot on" if b["exact_hit"] else "not too bad, called the result"
        crowd = (f" while {int(b['zeros'])} of the lads got handed their arse"
                 if b["zeros"] else "")
        lines.append(f"⭐ Best pick: **{b['pred_score']}** on {flag(b['home'])} "
                     f"{b['home']}–{b['away']} {flag(b['away'])} "
                     f"(FT {b['actual_score']}) — {verb}{crowd}.")

    # Stinkers — the one man missing while the rest of the league cashed in.
    strict, soft = stinkers(df, player)
    if len(strict):
        r = strict.sort_values("datetime").iloc[-1]
        pick = (f"**{r['home']} {r['pred_score']} {r['away']}**"
                if r["has_prediction"]
                else f"nothing on **{r['home']}–{r['away']}**")
        again = (f" — and not just once, {len(strict)} times"
                 if len(strict) > 1 else "")
        lines.append(f"🤢 Stinker: This innocent boy reckoned {pick} — every "
                     f"other sham said {r['actual_score']}{again}.")
    elif len(soft):
        r = soft.sort_values("datetime").iloc[-1]
        again = f" ({len(soft)} times)" if len(soft) > 1 else ""
        lines.append(f"🤢 Stinker: Blanked badly on **{r['home']} "
                     f"{r['actual_score']} {r['away']}** while everyone else "
                     f"scored{again}.")

    # Exact drought / never-scored watch. Two tiers: concern at 8, alarm at 10.
    hits = rows.index[rows["exact_hit"]]
    if len(hits) == 0 and len(rows) >= 5:
        lines.append(f"🥶 Still waiting on a **first exact score** after "
                     f"{len(rows)} games. Rooting for you.")
    elif len(hits):
        drought = len(rows) - 1 - hits[-1]
        if drought >= 10:
            lines.append(f"🪣 No exact in **{drought} games** — "
                         f"Not looking too good AT ALL")
        elif drought >= 8:
            lines.append(f"🥶 No exact in **{drought} games** — "
                         f"Not looking too good")

    # Form vs the field over the shared last-10 window.
    if s["last10"] == stats["last10"].max() and s["last10"] > 0:
        lines.append(f"🔥🐐 **On the Goats Milk** — {int(s['last10'])} pts "
                     f"from the last {LAST_N}.")
    elif s["last10"] == stats["last10"].min():
        lines.append(f"🧊 Shitest in the league: **{int(s['last10'])} pts** "
                     f"from the last {LAST_N}.")

    banker, bogey = team_form(df, player)
    if banker and banker[1] > 0:
        lines.append(f"🏦 Banker: {flag(banker[0])} **{banker[0]}** — "
                     f"{banker[1]:.1f} pts/game across {banker[2]}.")
    if bogey and bogey[1] < 0.5:
        lines.append(f"🥔 Blight: {flag(bogey[0])} **{bogey[0]}** — "
                     f"{bogey[1]:.1f} pts/game from {bogey[2]}. Look away.")

    # Blank watch.
    if s["blanks"]:
        lines.append(f"🪣 **{int(s['blanks'])} no-show"
                     f"{'s' if s['blanks'] != 1 else ''}** — Pull the Finger Out.")
    else:
        lines.append(f"📮 Little to be at — {int(s['played'])} from "
                     f"{int(s['played'])}. Professional.")
    return lines[:9]


# --------------------------------------------------------------------------- #
# FUT card + small HTML fragments
# --------------------------------------------------------------------------- #
CSS = """
<style>
.fut-wrap{display:flex;justify-content:center;padding:6px 0 2px;}
.fut{position:relative;width:272px;border-radius:20px;padding:20px 22px 14px;
  box-shadow:0 12px 30px rgba(30,58,95,.20);overflow:hidden;}
.fut-gold{background:linear-gradient(165deg,#FCE9A8 0%,#E9C84E 55%,#D4AF37 100%);color:#4a3a05;}
.fut-silver{background:linear-gradient(165deg,#F4F7FA 0%,#D9E0E8 55%,#B9C4D0 100%);color:#3a4654;}
.fut-bronze{background:linear-gradient(165deg,#F0D9C0 0%,#D9A876 55%,#B98046 100%);color:#4d3013;}
.fut-poopy{background:linear-gradient(165deg,#D9CBB5 0%,#A88B62 55%,#7B5B36 100%);color:#3d2c14;}
.fut .wm{position:absolute;right:-18px;bottom:-24px;font-size:7rem;opacity:.16;
  font-family:"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif;}
.fut .top{display:flex;justify-content:space-between;align-items:flex-start;}
.fut .ovr{font-size:2.9rem;font-weight:900;line-height:1;}
.fut .pos{font-size:1.05rem;font-weight:800;letter-spacing:.08em;margin-top:2px;}
.fut .meta{text-align:right;font-size:.8rem;font-weight:700;line-height:1.5;opacity:.85;}
.fut .name{font-size:1.5rem;font-weight:900;text-transform:uppercase;
  letter-spacing:.05em;text-align:center;margin:10px 0 8px;}
.fut hr{border:none;border-top:2px solid rgba(0,0,0,.18);margin:2px 0 10px;}
.fut .grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px 4px;text-align:center;}
.fut .grid b{font-size:1.15rem;font-weight:900;margin-right:4px;}
.fut .grid span{font-size:.78rem;font-weight:800;letter-spacing:.04em;opacity:.8;}
.fut .eyebrow{text-align:center;font-size:.66rem;font-weight:800;letter-spacing:.14em;
  text-transform:uppercase;opacity:.65;margin-top:12px;}
.fchips{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:4px 0 2px;}
.fchip{width:27px;height:27px;border-radius:8px;display:flex;align-items:center;
  justify-content:center;font-size:.75rem;font-weight:800;color:#fff;cursor:default;}
.f-x{background:#16A34A;}.f-o{background:#F59E0B;}.f-m{background:#DC2626;}
.f-b{background:#9CA3AF;}
.pctl{margin:8px 0;}
.pctl .lab{display:flex;justify-content:space-between;font-size:.86rem;
  font-weight:600;color:#374151;margin-bottom:3px;}
.pctl .lab .val{color:#6B7280;font-weight:700;}
.pctl .track{height:10px;border-radius:6px;background:#F3F4F6;overflow:hidden;}
.pctl .fill{height:100%;border-radius:6px;}
</style>
"""

# Position gag: best-to-worst rank mapped front-of-pitch to back.
_POSITIONS = ["ST", "CF", "CAM", "CM", "CDM", "CB", "GK"]
_TIER_LABEL = {"fut-gold": "Team of the Season", "fut-silver": "Rare Silver",
               "fut-bronze": "Bronze", "fut-poopy": "Poopy Edition"}


def _tier(rank: int, n: int) -> str:
    if rank == 1:
        return "fut-gold"
    if rank == n:
        return "fut-poopy"
    return "fut-silver" if rank <= 3 else "fut-bronze"


def fut_card_html(player: str, s: pd.Series, n_players: int,
                  banker_flag: str) -> str:
    tier = _tier(int(s["rank"]), n_players)
    pos = _POSITIONS[round((int(s["rank"]) - 1) / max(n_players - 1, 1)
                          * (len(_POSITIONS) - 1))]
    attrs = "".join(f"<div><b>{int(s[a])}</b><span>{a}</span></div>" for a in ATTRS)
    wm = "<div class='wm'>💩</div>" if tier == "fut-poopy" else ""
    return (f"<div class='fut-wrap'><div class='fut {tier}'>{wm}"
            f"<div class='top'><div><div class='ovr'>{int(s['rating'])}</div>"
            f"<div class='pos'>{pos}</div></div>"
            f"<div class='meta'>{banker_flag}<br>#{int(s['rank'])} of {n_players}</div></div>"
            f"<div class='name'>{_html.escape(player)}</div><hr>"
            f"<div class='grid'>{attrs}</div>"
            f"<div class='eyebrow'>{_TIER_LABEL[tier]}</div>"
            f"</div></div>")


def chips_html(chips: list[dict]) -> str:
    spans = "".join(f"<span class='fchip {c['cls']}' title=\"{_html.escape(c['tip'])}\">"
                    f"{c['lab']}</span>" for c in chips)
    return f"<div class='fchips'>{spans}</div>"


def _ordinal(n: int) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rank_bars_html(stats: pd.DataFrame, player: str) -> str:
    """League-rank bars: 1st fills the track, last barely registers."""
    s = stats.loc[player]
    n = len(stats)
    items = [("Points per game", "rk_ppg", f"{s['ppg']:.2f}"),
             ("Exact-score rate", "rk_acc", f"{s['acc_rate']:.0%}"),
             ("Correct-result rate", "rk_res", f"{s['res_rate']:.0%}"),
             ("Form (last 10 pts)", "rk_frm", f"{int(s['last10'])}"),
             ("Boldness", "rk_bold", f"{s['bold_rate']:.0%}"),
             ("Submission rate", "rk_rel", f"{s['rel_rate']:.0%}")]
    bars = []
    for label, col, val in items:
        rk = int(s[col])
        frac = (n - rk + 1) / n if n else 0
        colour = ("#16A34A" if frac >= 0.7
                  else "#F59E0B" if frac >= 0.4 else "#DC2626")
        bars.append(f"<div class='pctl'><div class='lab'><span>{label}</span>"
                    f"<span class='val'>{val} · {_ordinal(rk)} of {n}</span></div>"
                    f"<div class='track'><div class='fill' "
                    f"style='width:{frac * 100:.0f}%;background:{colour}'>"
                    f"</div></div></div>")
    return "".join(bars)


# --------------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------------- #
def radar_option(stats: pd.DataFrame, player: str, colour: str) -> dict:
    avg = stats[ATTRS].mean()
    return {
        "color": [colour, "#9CA3AF"],
        "legend": {"data": [player, "League avg"], "bottom": 0},
        "radar": {
            "indicator": [{"name": a, "max": 99} for a in ATTRS],
            "radius": "66%",
            "axisName": {"color": "#374151", "fontWeight": "bold"},
            "splitLine": {"lineStyle": {"color": "#E5E7EB"}},
            "splitArea": {"show": False},
            "axisLine": {"lineStyle": {"color": "#E5E7EB"}},
        },
        "series": [{"type": "radar", "data": [
            {"value": [int(stats.loc[player, a]) for a in ATTRS], "name": player,
             "areaStyle": {"opacity": 0.25}, "lineStyle": {"width": 3}},
            {"value": [round(float(avg[a]), 1) for a in ATTRS], "name": "League avg",
             "lineStyle": {"width": 1.5, "type": "dashed"}, "symbol": "none"},
        ]}],
    }


def radar_plotly(stats: pd.DataFrame, player: str, colour: str) -> go.Figure:
    avg = stats[ATTRS].mean()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[stats.loc[player, a] for a in ATTRS],
                                  theta=ATTRS, fill="toself", name=player,
                                  line=dict(color=colour, width=3)))
    fig.add_trace(go.Scatterpolar(r=[avg[a] for a in ATTRS], theta=ATTRS,
                                  name="League avg",
                                  line=dict(color="#9CA3AF", dash="dash")))
    fig.update_layout(height=340, polar=dict(radialaxis=dict(range=[0, 99])),
                      legend=dict(orientation="h", y=-0.15))
    return fig


# --------------------------------------------------------------------------- #
# Tab body
# --------------------------------------------------------------------------- #
def _hda_figure(df: pd.DataFrame, player: str, colour: str) -> go.Figure:
    """Predicted vs actual outcome mix (%) on the matches this player called."""
    rows = df[(df["player"] == player) & df["played"] & df["has_prediction"]]
    pred = np.sign(rows["pred_home"] - rows["pred_away"])
    act = np.sign(rows["actual_home"] - rows["actual_away"])
    cats = [1, 0, -1]
    pick = [float((pred == c).mean()) * 100 for c in cats]
    real = [float((act == c).mean()) * 100 for c in cats]
    fig = go.Figure([
        go.Bar(name="Picked", x=["Home win", "Draw", "Away win"], y=pick,
               marker_color=colour),
        go.Bar(name="Actually happened", x=["Home win", "Draw", "Away win"],
               y=real, marker_color="#9CA3AF"),
    ])
    fig.update_layout(height=300, barmode="group", yaxis_title="% of games",
                      legend=dict(orientation="h", y=1.15, x=0),
                      margin=dict(l=48, r=12, t=24, b=36))
    return fig


def _log_frame(df: pd.DataFrame, player: str) -> pd.DataFrame:
    rows = (df[(df["player"] == player) & df["played"]]
            .sort_values("datetime", ascending=False))
    return pd.DataFrame({
        "Date": pd.to_datetime(rows["datetime"]).dt.strftime("%-d %b"),
        "Stage": rows["stage_raw"].astype(str),
        # Tidy stage ("Group Stage", "Round of 16", …) drives the pills filter;
        # the visible Stage column keeps the specific group/leg label.
        "_stage": rows["stage"].astype(str),
        "Fixture": [f"{flag(h)} {h} {s} {a} {flag(a)}"
                    for h, a, s in zip(rows["home"], rows["away"],
                                       rows["actual_score"])],
        "Pick": [str(p) if hp else "— no pick"
                 for p, hp in zip(rows["pred_score"], rows["has_prediction"])],
        "Pts": rows["points"].astype(int),
        "_k": np.select(
            [~rows["has_prediction"], rows["exact_hit"], rows["outcome_only"]],
            ["blank", "exact", "result"], default="miss"),
    })


def _log_row_style(row: pd.Series) -> list[str]:
    bg = {"exact": "#DCFCE7", "result": "#FEF3C7",
          "blank": "#F3F4F6", "miss": "#FEE2E2"}[row["_k"]]
    return [f"background-color:{bg}"] * len(row)


def render(df: pd.DataFrame, players: list[str], cmap: dict[str, str]) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    stats = player_stats(df)

    # Picker in standings order, defaulting to the leader (ego demands it).
    # stats is indexed by player: alphabetical first, then a stable sort by
    # points, so tied players keep name order.
    order = list(stats.sort_index()
                 .sort_values("pts", ascending=False, kind="stable").index)
    # Seed the widget from the URL so the selection survives the live-window
    # auto page reload (a reload starts a fresh Streamlit session, wiping
    # normal widget state). The widget takes over from session_state after.
    if "player_pick" not in st.session_state:
        qp = st.query_params.get("player", "")
        st.session_state["player_pick"] = qp if qp in order else order[0]
    picked = st.segmented_control("Player", order, key="player_pick",
                                  label_visibility="collapsed")
    player = picked or order[0]  # segmented_control returns None on deselect
    st.query_params["player"] = player
    s = stats.loc[player]

    # ---- Hero: FUT card + headline metrics + form chips ------------------- #
    banker, bogey = team_form(df, player)
    left, right = st.columns([1, 1.9], gap="large")
    with left:
        st.markdown(fut_card_html(player, s, len(players),
                                  flag(banker[0]) if banker else "⚽"),
                    unsafe_allow_html=True)
    with right:
        prev_rank = rank_before_latest(df)
        move = (int(prev_rank[player]) - int(s["rank"])
                if player in prev_rank.index else 0)
        gap = int(s["pts"]) - int(stats["pts"].max())
        second_pts = int(stats.loc[stats.index != player, "pts"].max())

        m1, m2, m3 = st.columns(3)
        m1.metric("Rank", f"#{int(s['rank'])}",
                  delta=(f"{move:+d} places" if move else None), border=True)
        m2.metric("Points", int(s["pts"]),
                  delta=(f"+{int(s['pts']) - second_pts} clear" if s["rank"] == 1
                         else f"{gap} behind leader"), border=True)
        m3.metric("Exact scores", int(s["exacts"]),
                  help="3-pointers — score dead right", border=True)
        m4, m5, m6 = st.columns(3)
        m4.metric("Correct results", int(s["results"]),
                  help="Right outcome, wrong score (1 pt)", border=True)
        m5.metric("Misses", int(s["played"] - s["exacts"] - s["results"]
                                - s["blanks"]), border=True)
        m6.metric("Blanks", int(s["blanks"]),
                  help="No prediction submitted", border=True)

        st.caption(f"Form — last {LAST_N} (oldest → newest) · "
                   "🟩 exact 🟨 result 🟥 miss ⬜ blank")
        st.markdown(chips_html(form_chips(df, player)), unsafe_allow_html=True)

    # ---- Scouting report --------------------------------------------------- #
    try:
        from streamlit_extras.stylable_container import stylable_container
        card = stylable_container(key="scout_card", css_styles="""
            {
                border: 1px solid #E6EAF0;
                border-left: 6px solid #1E3A5F;
                border-radius: 14px;
                padding: 14px 20px;
                box-shadow: 0 6px 18px rgba(30,58,95,0.07);
            }""")
    except Exception:
        card = st.container(border=True)
    with card:
        st.markdown(f"#### 🕵️ Scouting report — {player}")
        st.markdown("  \n".join(scouting_report(df, stats, player)))

    # ---- Attributes: radar vs league + percentile bars --------------------- #
    st.divider()
    rcol, pcol = st.columns([1.1, 1], gap="large")
    with rcol:
        st.markdown("**:material/radar: Attribute profile**")
        if _HAS_ECHARTS:
            st_echarts(radar_option(stats, player, cmap[player]),
                       height="360px", key=f"radar-{player}")
        else:
            st.plotly_chart(radar_plotly(stats, player, cmap[player]),
                            use_container_width=True)
        st.caption(" · ".join(f"**{a}**{ATTR_HELP[a].split('—')[1]}"
                              for a in ATTRS))
    with pcol:
        st.markdown("**:material/format_list_numbered: League ranks**")
        st.markdown(rank_bars_html(stats, player), unsafe_allow_html=True)
        st.caption("Rank vs the league per stat — longer bar, higher rank. "
                   "Skill rates exclude blank picks.")

    # ---- Tendencies --------------------------------------------------------- #
    st.divider()
    st.markdown("**:material/psychology: Tendencies**")
    t1, t2, t3 = st.columns([1, 1.3, 1], gap="large")
    with t1:
        st.caption("Go-to scorelines")
        habit = scoreline_habits(df, player)
        if habit.empty:
            st.info("No submitted picks yet.")
        else:
            st.dataframe(habit, hide_index=True, width="stretch")
    with t2:
        st.caption("Picked vs what actually happened")
        st.plotly_chart(_hda_figure(df, player, cmap[player]),
                        use_container_width=True)
    with t3:
        st.caption("Banker & bogey teams (3+ games)")
        if banker:
            st.metric(f"🏦 {flag(banker[0])} {banker[0]}",
                      f"{banker[1]:.1f} pts/gm", delta=f"{banker[2]} games",
                      delta_color="off", border=True)
        if bogey:
            st.metric(f"🥔 {flag(bogey[0])} {bogey[0]}",
                      f"{bogey[1]:.1f} pts/gm", delta=f"{bogey[2]} games",
                      delta_color="off", border=True)

    # ---- Match log ---------------------------------------------------------- #
    st.divider()
    st.markdown("**:material/receipt_long: Prediction log**")
    log = _log_frame(df, player)
    # dict.fromkeys = dedupe preserving play order (newest first in the log,
    # so reverse to read Group Stage -> Final left to right).
    stages = list(dict.fromkeys(log["_stage"]))[::-1]
    pick_stage = st.pills("Stage", ["All"] + stages, default="All",
                          key="log_stage", label_visibility="collapsed")
    if pick_stage and pick_stage != "All":
        log = log[log["_stage"] == pick_stage]
    st.dataframe(log.style.apply(_log_row_style, axis=1),
                 hide_index=True, width="stretch", height=420,
                 column_config={"_k": None, "_stage": None})
