"""Regression tests for live-score handling in the feed parser.

Bug: an in-progress game (`finished=FALSE`, `time_elapsed="live"`) was marked
not-played because the code used the `finished` flag as the `played` flag, so
its live score rendered as "-". Meanwhile every future fixture arrives as a
0-0 row, so we must NOT treat a bare 0-0 as played either. `_is_started`
distinguishes them via `time_elapsed`.

Run: ./venv/bin/python -m pytest test_live_scores.py -q
"""
import data_loader as dl

LIVE = {"home_team_name_en": "Netherlands", "away_team_name_en": "Sweden",
        "home_score": "2", "away_score": "0",
        "finished": "FALSE", "time_elapsed": "live"}
FUTURE = {"home_team_name_en": "Japan", "away_team_name_en": "Sweden",
          "home_score": "0", "away_score": "0",
          "finished": "FALSE", "time_elapsed": "notstarted"}
FINISHED = {"home_team_name_en": "Netherlands", "away_team_name_en": "Japan",
            "home_score": "2", "away_score": "2",
            "finished": "TRUE", "time_elapsed": "finished"}


def test_live_game_is_played_and_live_with_score():
    # (home, away, played, live). Keys are casefolded join keys (see canon).
    parsed = dl._parse_games([LIVE])
    assert parsed[(dl.canon("Netherlands"), dl.canon("Sweden"))] == (2, 0, True, True)


def test_future_game_is_not_played_nor_live():
    parsed = dl._parse_games([FUTURE])
    assert parsed[(dl.canon("Japan"), dl.canon("Sweden"))] == (0, 0, False, False)


def test_finished_game_is_played_not_live():
    parsed = dl._parse_games([FINISHED])
    hs, aw, played, live = parsed[(dl.canon("Netherlands"), dl.canon("Japan"))]
    assert played is True and live is False


def test_missing_time_elapsed_falls_back_to_finished_flag():
    g = {**LIVE, "finished": "TRUE"}
    del g["time_elapsed"]
    played, live = dl._parse_games([g])[(dl.canon("Netherlands"), dl.canon("Sweden"))][2:]
    assert played is True and live is False   # finished, not live
    g2 = {**FUTURE}
    del g2["time_elapsed"]
    assert dl._parse_games([g2])[(dl.canon("Japan"), dl.canon("Sweden"))][2:] == (False, False)


def test_placeholder_and_nonnumeric_rows_skipped():
    rows = [
        {"home_team_name_en": "", "away_team_name_en": "X",
         "home_score": "1", "away_score": "0", "finished": "TRUE"},
        {"home_team_name_en": "A", "away_team_name_en": "B",
         "home_score": None, "away_score": None, "finished": "FALSE",
         "time_elapsed": "notstarted"},
    ]
    assert dl._parse_games(rows) == {}


def test_canon_is_case_and_space_insensitive():
    # The Sheet is inconsistent with itself: "Ivory Coast" on some rows,
    # "Ivory coast " (lower c, trailing space) on others. Both must produce the
    # same join key, or the finished score never reaches that fixture.
    assert dl.canon("Ivory coast ") == dl.canon("Ivory Coast")


def test_join_key_survives_sheet_casing_inconsistency():
    # API spells it "Ivory Coast"; the Sheet row for that fixture says
    # "Ivory coast ". The Sheet-side lookup key must hit the API-side key.
    api = dl._parse_games([{**FINISHED, "home_team_name_en": "Ivory Coast",
                            "away_team_name_en": "Norway"}])
    sheet_key = (dl.canon("Ivory coast "), dl.canon("Norway "))
    assert sheet_key in api


def test_et_match_day_straddles_uk_dates():
    import pandas as pd
    # UK 01:30 on 20 Jun is the prior ET evening (19 Jun); UK 18:00 stays 20 Jun.
    md = dl.et_match_day(pd.Series(pd.to_datetime(
        ["2026-06-20 01:30", "2026-06-20 18:00"])))
    assert str(md.iloc[0].date()) == "2026-06-19"
    assert str(md.iloc[1].date()) == "2026-06-20"
