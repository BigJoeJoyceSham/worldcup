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
    # (home, away, played, live, final). Keys are casefolded join keys (canon).
    parsed = dl._parse_games([LIVE])
    assert parsed[(dl.canon("Netherlands"), dl.canon("Sweden"))] == (2, 0, True, True, None)


def test_future_game_is_not_played_nor_live():
    parsed = dl._parse_games([FUTURE])
    assert parsed[(dl.canon("Japan"), dl.canon("Sweden"))] == (0, 0, False, False, None)


def test_finished_game_is_played_not_live():
    parsed = dl._parse_games([FINISHED])
    hs, aw, played, live, final = parsed[(dl.canon("Netherlands"), dl.canon("Japan"))]
    assert played is True and live is False


def test_missing_time_elapsed_falls_back_to_finished_flag():
    g = {**LIVE, "finished": "TRUE"}
    del g["time_elapsed"]
    played, live = dl._parse_games([g])[(dl.canon("Netherlands"), dl.canon("Sweden"))][2:4]
    assert played is True and live is False   # finished, not live
    g2 = {**FUTURE}
    del g2["time_elapsed"]
    assert dl._parse_games([g2])[(dl.canon("Japan"), dl.canon("Sweden"))][2:4] == (False, False)


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


def test_sheet_nicknames_join_to_feed_full_names():
    # The Sheet abbreviates some teams; those must map to the feed's full name
    # or a live score never reaches the fixture (e.g. Swiss vs Algeria R32).
    assert dl.canon("Swiss") == dl.canon("Switzerland")
    assert dl.canon("Aussies") == dl.canon("Australia")
    assert dl.canon("Bosnia") == dl.canon("Bosnia and Herzegovina")


def test_join_key_survives_sheet_casing_inconsistency():
    # API spells it "Ivory Coast"; the Sheet row for that fixture says
    # "Ivory coast ". The Sheet-side lookup key must hit the API-side key.
    api = dl._parse_games([{**FINISHED, "home_team_name_en": "Ivory Coast",
                            "away_team_name_en": "Norway"}])
    sheet_key = (dl.canon("Ivory coast "), dl.canon("Norway "))
    assert sheet_key in api


def test_display_spelling_prefers_majority_over_typo():
    # "Ivory Coast" appears 3×, the typo "Ivory coast " once -> majority wins.
    names = ["Ivory Coast", "Ivory Coast", "Ivory Coast", "Ivory coast "]
    spell = dl._display_spellings(names)
    assert spell[dl.canon("Ivory coast ")] == "Ivory Coast"


def test_display_spelling_prefers_alias_pretty_form():
    # An aliased team always renders as its pretty canonical form.
    spell = dl._display_spellings(["turkey", "Türkiye"])
    assert spell[dl.canon("turkey")] == "Türkiye"


BELGIUM = {  # 2-2 at full time; Tielemans' 125' ET penalty made it 3-2.
    "home_team_name_en": "Belgium", "away_team_name_en": "Senegal",
    "home_score": "3", "away_score": "2",
    "home_scorers": '{"Romelu Lukaku 86\'","Youri Tielemans 89\'","Youri Tielemans 125(P)\'"}',
    "away_scorers": '{"Habib Diarra 25\'","Ismaïla Sarr 51\'"}',
    "finished": "TRUE", "time_elapsed": "finished"}

PENS = {  # 1-1 at full time, decided on penalties (kept in a separate field).
    "home_team_name_en": "Germany", "away_team_name_en": "Paraguay",
    "home_score": "1", "away_score": "1",
    "home_scorers": '{"Kai Havertz 54\'"}', "away_scorers": '{"Khvliv Ansisv 42\'"}',
    "home_penalty_score": "3", "away_penalty_score": "4",
    "finished": "TRUE", "time_elapsed": "finished"}


def test_extra_time_goal_dropped_to_ninety_minute_score():
    # League scores on the 90' result: Belgium's 125' ET goal must not count,
    # so the fixture reads 2-2 (a draw), not the feed's 3-2.
    parsed = dl._parse_games([BELGIUM])
    hs, aw, played, live, final = parsed[(dl.canon("Belgium"), dl.canon("Senegal"))]
    assert (hs, aw) == (2, 2) and played is True and live is False


def test_extra_time_win_carries_final_result_and_label():
    # The after-90 result is preserved so the UI can badge it: Belgium won 3-2
    # in extra time (no penalties).
    final = dl._parse_games([BELGIUM])[(dl.canon("Belgium"), dl.canon("Senegal"))][4]
    assert final == (3, 2, None, None)
    assert dl._final_label(final) == "3-2 AET"


def test_penalty_win_carries_final_result_and_label():
    final = dl._parse_games([PENS])[(dl.canon("Germany"), dl.canon("Paraguay"))][4]
    assert final == (1, 1, 3, 4)
    assert dl._final_label(final) == "3-4 pens"


def test_decided_in_ninety_has_no_final_result():
    # A game with a winner inside 90' carries no after-90 badge.
    g = {"home_team_name_en": "E", "away_team_name_en": "F",
         "home_score": "2", "away_score": "1",
         "home_scorers": '{"X 30\'","Y 70\'"}', "away_scorers": '{"Z 60\'"}',
         "finished": "TRUE", "time_elapsed": "finished"}
    final = dl._parse_games([g])[(dl.canon("E"), dl.canon("F"))][4]
    assert final is None and dl._final_label(final) == ""


def test_swap_final_flips_home_away_and_pens():
    assert dl._swap_final((3, 2, None, None)) == (2, 3, None, None)
    assert dl._swap_final((1, 1, 3, 4)) == (1, 1, 4, 3)
    assert dl._swap_final(None) is None


def test_stoppage_time_goal_counts_as_ninety_minute():
    # "90+1'" is regulation stoppage, not extra time -> it counts.
    g = {**PENS, "away_scorers": '{"Issa Diop 90+1\'"}'}
    assert dl._parse_games([g])[(dl.canon("Germany"), dl.canon("Paraguay"))][:2] == (1, 1)


def test_full_time_stoppage_winner_counts():
    # A 90+7' goal is second-half stoppage (part of full time), not extra time
    # (which the feed writes as raw minutes > 90). It must count: stays 2-1.
    g = {"home_team_name_en": "A", "away_team_name_en": "B",
         "home_score": "2", "away_score": "1",
         "home_scorers": '{"X 30\'","Y 90+7\'"}', "away_scorers": '{"Z 60\'"}',
         "finished": "TRUE", "time_elapsed": "finished"}
    assert dl._parse_games([g])[(dl.canon("A"), dl.canon("B"))][:2] == (2, 1)


def test_extra_time_goal_by_raw_minute_dropped():
    # 1-1 at 90, then a 105' extra-time goal (raw cumulative minute) makes the
    # feed read 2-1 -> the 90-minute result is the 1-1 draw.
    g = {"home_team_name_en": "C", "away_team_name_en": "D",
         "home_score": "2", "away_score": "1",
         "home_scorers": '{"P 20\'","Q 105\'"}', "away_scorers": '{"R 80\'"}',
         "finished": "TRUE", "time_elapsed": "finished"}
    assert dl._parse_games([g])[(dl.canon("C"), dl.canon("D"))][:2] == (1, 1)


def test_penalty_shootout_keeps_ninety_minute_draw():
    # home_score/away_score already exclude shootout goals -> untouched 1-1.
    assert dl._parse_games([PENS])[(dl.canon("Germany"), dl.canon("Paraguay"))][:2] == (1, 1)


def test_unreconciled_scorers_fall_back_to_reported_score():
    # If scorer lists don't match the reported score (missing/malformed data),
    # trust the feed's score rather than silently under/over-counting.
    g = {**BELGIUM, "home_scorers": "null", "away_scorers": "null"}
    assert dl._parse_games([g])[(dl.canon("Belgium"), dl.canon("Senegal"))][:2] == (3, 2)


def test_merge_live_game_always_comes_from_feed():
    # A live game must show the feed's live score even if the Sheet holds a
    # (stale) value -- live updates are the whole point of the dashboard.
    hit = (1, 0, True, True, None)  # feed: home leading, in progress
    assert dl._merge_fixture(9, 9, True, hit) == (1, 0, True, True, None)


def test_merge_finished_prefers_sheet_score_with_feed_badge():
    # Finished and entered in the Sheet: the Sheet wins the score, the feed only
    # contributes the after-90 badge (Belgium 2-2 on the Sheet, 3-2 AET on feed).
    hit = (2, 2, True, False, (3, 2, None, None))
    assert dl._merge_fixture(2, 2, True, hit) == (2, 2, True, False, (3, 2, None, None))


def test_merge_just_finished_not_yet_in_sheet_uses_feed():
    # Sheet blank (nobody's typed it in yet) -> the feed fills the result.
    hit = (3, 0, True, False, None)
    assert dl._merge_fixture(None, None, False, hit) == (3, 0, True, False, None)


def test_merge_no_feed_row_keeps_sheet():
    # Feed miss (alias gap / TBD knockout) must never wipe a Sheet result.
    assert dl._merge_fixture(2, 1, True, None) == (2, 1, True, False, None)
    assert dl._merge_fixture(None, None, False, None) == (None, None, False, False, None)


def test_et_match_day_straddles_uk_dates():
    import pandas as pd
    # UK 01:30 on 20 Jun is the prior ET evening (19 Jun); UK 18:00 stays 20 Jun.
    md = dl.et_match_day(pd.Series(pd.to_datetime(
        ["2026-06-20 01:30", "2026-06-20 18:00"])))
    assert str(md.iloc[0].date()) == "2026-06-19"
    assert str(md.iloc[1].date()) == "2026-06-20"
