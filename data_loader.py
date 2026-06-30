"""Data loading + tidy-frame construction for the World Cup predictions dashboard.

Source of truth is the shared Google Sheet, read live via its public xlsx
export URL. Each player has their own sheet (predicted scores + computed
points); the `Results` sheet holds actual scores. We join them by row
position (the sheet is built so player row N aligns to Results row N) into a
single tidy long table: one row per (player x match).

Later, a live score feed (API / scrape) can replace the `Results` half
without touching anything downstream of `load_long()`.
"""
from __future__ import annotations

import io
import time
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests

SHEET_ID = "13vUz3Z8Tk2j-rq5Ma36eRlrsc8VnGUYK0CK_wMbPxi0"
EXPORT_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=xlsx"

# Live results feed (open, no auth). Carries real-time scores keyed by English
# team names; we join it to the Sheet's fixtures by (home, away) name pair.
API_URL = "https://worldcup26.ir/get/games"

# Team-name aliases -> canonical form. The Sheet and the API spell several
# teams differently (and the Sheet is even inconsistent with itself), so we
# normalise both sides through this map before joining.
_ALIASES = {
    "czechia": "Czech Republic", "czech republic": "Czech Republic",
    "congo dr": "DR Congo", "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "turkiye": "Türkiye", "türkiye": "Türkiye", "turkey": "Türkiye",
    "usa": "USA", "united states": "USA",
}


def canon(name) -> str:
    """Normalise a team name to a canonical, case-insensitive join key.

    Known spelling differences go through _ALIASES; the result is then casefolded
    so any leftover case- or whitespace-only inconsistency still joins (the Sheet
    is inconsistent even with itself, e.g. "Ivory Coast" vs "Ivory coast "). Used
    only to build join keys on both sides — never for display.
    """
    s = str(name).strip()
    return _ALIASES.get(s.lower(), s).casefold()


def _display_spellings(names) -> dict[str, str]:
    """Map each canon join-key -> one tidy display spelling for a set of names.

    The Sheet is hand-typed and inconsistent (e.g. "Ivory coast " vs "Ivory
    Coast"). For each team we pick the pretty _ALIASES value when one exists,
    otherwise the most common stripped spelling actually seen — so a one-off
    typo yields to the majority spelling. Display only; joins use canon()."""
    from collections import Counter
    pretty = {v.casefold(): v for v in _ALIASES.values()}
    counts: dict[str, Counter] = {}
    for n in names:
        counts.setdefault(canon(n), Counter())[str(n).strip()] += 1
    return {key: pretty.get(key, c.most_common(1)[0][0])
            for key, c in counts.items()}


# Fixture clocks are stored/displayed in UK local time, but the tournament is in
# the US, so a "match day" is a 24h calendar day in New York time. Converting
# UK->ET means a match day can straddle two UK dates (e.g. a UK 01:30 kickoff is
# the previous ET evening) — that's intended.
DISPLAY_TZ = ZoneInfo("Europe/London")
MATCHDAY_TZ = ZoneInfo("America/New_York")


def et_match_day(dt_series) -> pd.Series:
    """Map UK-local kickoff datetimes to their New York match-day date (naive,
    normalised to midnight ET)."""
    uk = pd.to_datetime(dt_series).dt.tz_localize(
        DISPLAY_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return uk.dt.tz_convert(MATCHDAY_TZ).dt.tz_localize(None).dt.normalize()


def _is_finished(value) -> bool:
    """Treat the feed's `finished` field as done whether it arrives as the
    string "TRUE", a JSON boolean, or 1 — so a format change can't silently
    flip every game to 'not played' and zero the standings."""
    return str(value).strip().lower() in {"true", "1", "yes", "finished"}


def _is_started(game) -> bool:
    """True when a game has a real score to show — it's either finished OR
    currently in progress. The feed returns *every* future fixture as a 0-0
    row (`finished=FALSE`, `time_elapsed="notstarted"`), so we must NOT treat a
    bare 0-0 as played, or all 70+ upcoming games would post as scored draws.
    The discriminator is `time_elapsed`: "live" means in progress, "notstarted"
    means scheduled. We fall back to the `finished` flag if the field is absent."""
    if _is_finished(game.get("finished")):
        return True
    return str(game.get("time_elapsed", "")).strip().lower() == "live"


def _is_live(game) -> bool:
    """In progress right now: started but not finished."""
    return (not _is_finished(game.get("finished"))
            and str(game.get("time_elapsed", "")).strip().lower() == "live")


def _parse_games(games) -> dict[tuple[str, str], tuple[int, int, bool, bool]]:
    """Pure parse of the feed's `games` list -> {(home, away): (home_score,
    away_score, played, live)}, keyed by canonical team names. `played` is True
    for finished AND in-progress games so live scores surface; `live` flags the
    in-progress ones so the UI can show "LIVE" rather than "FT". Placeholder
    knockout rows (no team name) and non-numeric scores are skipped."""
    out: dict[tuple[str, str], tuple[int, int, bool, bool]] = {}
    for g in games:
        h, a = g.get("home_team_name_en"), g.get("away_team_name_en")
        if not h or not a:
            continue
        try:
            hs, aw = int(g["home_score"]), int(g["away_score"])
        except (TypeError, ValueError, KeyError):
            continue
        out[(canon(h), canon(a))] = (hs, aw, _is_started(g), _is_live(g))
    return out


def fetch_api_results(url: str = API_URL, attempts: int = 5
                      ) -> dict[tuple[str, str], tuple[int, int, bool, bool]]:
    """Pull live results from the feed -> {(home, away): (home_score, away_score,
    played)}, keyed by canonical team names.

    The feed is flaky: ~70% of single requests die mid-handshake (SSL EOF /
    connection reset). A single failure here makes `load_long` fall back to the
    stale Sheet, so live scores appear frozen. We retry with short backoff to
    push effective reliability above 99%; only a full miss raises."""
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                                timeout=30)
            resp.raise_for_status()
            return _parse_games(resp.json().get("games", []))
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(0.5 * (i + 1))
    raise last_exc  # type: ignore[misc]

# Sheet tab names for each player. Note 'Noonan ' has a trailing space in the
# source workbook; we keep the exact key for reading and strip for display.
PLAYERS = ["Daithi", "Kian", "Mick", "Owen", "Reddy", "Garret", "Mac", "Noonan "]

# Player-sheet columns (header=None, 0-indexed). Row 0 is blank; matches start
# at row 1 and align to Results row 0.
_P_COLS = {
    0: "matchday", 1: "date", 2: "time", 3: "home", 4: "pred_home",
    5: "pred_away", 6: "away", 7: "pred_diff", 8: "outcome_pt",
    9: "exact_pt", 10: "pred_score", 11: "points",
}
# Results columns (header=None, 0-indexed).
_R_COLS = {
    0: "matchday", 1: "date", 2: "time", 3: "stage_raw", 4: "home",
    5: "actual_home", 6: "actual_away", 7: "away", 8: "actual_score",
    9: "actual_diff",
}


def stage_of(raw: str) -> str:
    """Collapse the raw stage label ('Group A', 'Round of 16 (R3)') into a tidy
    category used for grouping and the per-stage breakdown."""
    s = str(raw)
    if s.startswith("Group"):
        return "Group Stage"
    for key in ("Round of 32", "Round of 16", "Quarter-final",
                "Semi-final", "Third-place", "Final"):
        if s.startswith(key):
            return key.replace("Third-place", "Third Place")
    return s


def _read_workbook(source: str | bytes | None) -> bytes:
    """Return raw xlsx bytes from pre-fetched bytes, a local path, or the live
    export URL. Accepting bytes lets callers download the workbook once and pass
    it to multiple load_long() calls instead of re-fetching each time."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if source and not str(source).startswith("http") and source != "live":
        with open(source, "rb") as fh:
            return fh.read()
    resp = requests.get(EXPORT_URL, timeout=30)
    resp.raise_for_status()
    return resp.content


def load_long(source: str | None = "live", use_api: bool = True) -> pd.DataFrame:
    """Build the tidy long table: one row per (player, match).

    Pass a local .xlsx path for offline use, or "live"/None to pull the
    current Google Sheet for predictions + fixtures.

    When ``use_api`` is set (default), actual scores come from the live results
    feed (the source of truth going forward) and points are recomputed from
    each prediction against the feed's result. If the feed is unreachable we
    silently fall back to the scores recorded in the Sheet.
    """
    raw = _read_workbook(source)
    # Parse the workbook ONCE. Calling pd.read_excel per sheet re-parses the
    # whole file each time (openpyxl), which was ~13s for 9 sheets; ExcelFile
    # loads it a single time and every .parse() reuses that.
    xls = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")

    results = xls.parse("Results", header=None).rename(columns=_R_COLS)
    results["match_id"] = range(1, len(results) + 1)
    results["stage"] = results["stage_raw"].map(stage_of)
    results["played"] = results["actual_home"].notna() & results["actual_away"].notna()
    results["datetime"] = pd.to_datetime(
        results["date"].astype(str).str.slice(0, 10) + " "
        + results["time"].astype(str)
    )

    res_keys = results[[
        "match_id", "datetime", "date", "stage", "stage_raw", "home", "away",
        "actual_home", "actual_away", "actual_score", "played",
    ]]

    frames = []
    for player in PLAYERS:
        p = xls.parse(player, header=None).rename(columns=_P_COLS)
        # Drop the blank leading row; remaining rows align 1:1 with Results.
        p = p.iloc[1:].reset_index(drop=True)
        p = p.iloc[: len(results)].copy()
        p["match_id"] = range(1, len(p) + 1)
        p["player"] = player.strip()
        keep = ["match_id", "player", "pred_home", "pred_away", "pred_score",
                "outcome_pt", "exact_pt", "points"]
        frames.append(p[keep])

    long = pd.concat(frames, ignore_index=True).merge(res_keys, on="match_id")

    # Tidy displayed team names so a hand-typed Sheet inconsistency (e.g.
    # "Ivory coast " on one fixture) renders as the team's majority spelling.
    # Join keys still go through canon(), so this is purely cosmetic.
    spell = _display_spellings(pd.concat([long["home"], long["away"]]))
    long["home"] = long["home"].map(lambda n: spell.get(canon(n), str(n).strip()))
    long["away"] = long["away"].map(lambda n: spell.get(canon(n), str(n).strip()))

    for col in ("points", "exact_pt", "outcome_pt", "pred_home", "pred_away",
                "actual_home", "actual_away"):
        long[col] = pd.to_numeric(long[col], errors="coerce")

    # ET match day (24h NY window) for every fixture; `live` only the API sets.
    long["match_day"] = et_match_day(long["datetime"])
    long["live"] = False

    results_origin = "Google Sheet (manual)"
    if use_api:
        try:
            api = fetch_api_results()
        except Exception as exc:  # noqa: BLE001 - degrade to Sheet scores
            api = None
            results_origin = f"Sheet fallback (API failed: {exc})"
        if api:
            # Override actuals from the API where it has the fixture; otherwise
            # KEEP whatever the Sheet recorded. A join miss (alias gap, knockout
            # TBD, swapped orientation) must not wipe a real result to unplayed.
            keys = zip(long["home"].map(canon), long["away"].map(canon),
                       long["actual_home"], long["actual_away"], long["played"])
            ah, aw, pl, lv = [], [], [], []
            for k_home, k_away, s_home, s_away, s_played in keys:
                hit = api.get((k_home, k_away))
                if hit is not None:
                    ah.append(hit[0]); aw.append(hit[1])
                    pl.append(hit[2]); lv.append(hit[3])
                    continue
                # The API and Sheet don't always agree on which side is home,
                # so also try the swapped pairing and flip the scores back.
                swapped = api.get((k_away, k_home))
                if swapped is not None:
                    ah.append(swapped[1]); aw.append(swapped[0])
                    pl.append(swapped[2]); lv.append(swapped[3])
                else:
                    ah.append(s_home); aw.append(s_away)
                    pl.append(bool(s_played)); lv.append(False)
            long["actual_home"], long["actual_away"] = ah, aw
            long["played"], long["live"] = pl, lv
            results_origin = "live API (worldcup26.ir), Sheet for unmatched"

    # Recompute points from each prediction against the (possibly API) result,
    # so totals always reflect the current source of truth.
    #
    # Note: the Sheet scores a *blank* prediction as 0-0 (so a no-show earns a
    # point on any drawn game). We replicate that to stay consistent with the
    # standings the league is actually playing by -- hence pred.fillna(0) for
    # scoring -- but keep `has_prediction` to exclude blanks from tendency stats.
    has_pred = long["pred_home"].notna() & long["pred_away"].notna()
    pred_h = long["pred_home"].fillna(0)
    pred_a = long["pred_away"].fillna(0)
    has_act = long["actual_home"].notna() & long["actual_away"].notna() & long["played"]
    exact = has_act & (pred_h == long["actual_home"]) & (pred_a == long["actual_away"])
    same_outcome = has_act & (
        np.sign(pred_h - pred_a)
        == np.sign(long["actual_home"] - long["actual_away"]))
    long["exact_pt"] = np.where(exact, 3, 0)
    long["outcome_pt"] = np.where(same_outcome, 1, 0)
    long["points"] = np.maximum(long["exact_pt"], long["outcome_pt"])
    long["actual_score"] = [
        f"{int(h)}-{int(a)}" if pd.notna(h) and pd.notna(a) and pl else "-"
        for h, a, pl in zip(long["actual_home"], long["actual_away"], long["played"])]

    long["has_prediction"] = has_pred
    long["exact_hit"] = long["played"] & (long["exact_pt"] == 3)
    long["outcome_only"] = long["played"] & (long["exact_pt"] != 3) & (long["outcome_pt"] == 1)
    long["miss"] = long["played"] & (long["points"] == 0)
    long.attrs["results_origin"] = results_origin
    return long.sort_values(["datetime", "match_id", "player"]).reset_index(drop=True)


def players(long: pd.DataFrame) -> list[str]:
    return sorted(long["player"].unique())


if __name__ == "__main__":
    df = load_long("wc2026.xlsx")
    print("rows:", len(df), "| players:", df['player'].nunique(),
          "| matches:", df['match_id'].nunique(),
          "| played:", df[df.player == 'Kian'].played.sum())
    tot = (df[df.played].groupby("player")["points"].sum()
           .sort_values(ascending=False))
    print("\nLeaderboard (played only):")
    print(tot.to_string())
    print("\nExact-score hits per player:")
    print(df[df.exact_hit].groupby("player").size().sort_values(ascending=False).to_string())
