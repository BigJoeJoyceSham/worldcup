"""Presentation helpers for the WC 2026 Pool dashboard, kept out of app.py:

  * setup_plotly_theme() - one house style for every Plotly figure
  * build_recap()        - a generated "Matchday N" recap (pure pandas)

No Streamlit import at module load, so these stay unit-testable headless.
"""
from __future__ import annotations

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
