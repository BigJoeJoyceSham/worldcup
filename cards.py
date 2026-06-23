"""Card styling for the WC 2026 Pool dashboard.

Pure presentation helpers that turn the tidy long-frame into the clean,
card-based look from the reference apps:

  * match_card_html()  - a fixture card: flags + teams + a green/amber/red
                         split of the friends' picks + per-player pick chips.
  * stat_cards_html()  - tidy big-number cards (label / value / delta-vs-field).

Everything returns an HTML string for st.markdown(..., unsafe_allow_html=True);
no Streamlit import here so it stays unit-testable.
"""
from __future__ import annotations

import html as _html

import pandas as pd

# --------------------------------------------------------------------------- #
# Flags (emoji render fine in the browser, unlike in Pillow)
# --------------------------------------------------------------------------- #
_FLAGS = {
    "algeria": "🇩🇿", "argentina": "🇦🇷", "australia": "🇦🇺", "austria": "🇦🇹",
    "belgium": "🇧🇪", "bosnia and herzegovina": "🇧🇦", "brazil": "🇧🇷",
    "canada": "🇨🇦", "cape verde": "🇨🇻", "colombia": "🇨🇴",
    "congo dr": "🇨🇩", "dr congo": "🇨🇩", "croatia": "🇭🇷", "curaçao": "🇨🇼",
    "czech republic": "🇨🇿", "czechia": "🇨🇿", "ecuador": "🇪🇨", "egypt": "🇪🇬",
    "england": "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "france": "🇫🇷", "germany": "🇩🇪", "ghana": "🇬🇭", "haiti": "🇭🇹",
    "iran": "🇮🇷", "iraq": "🇮🇶", "ivory coast": "🇨🇮", "côte d'ivoire": "🇨🇮",
    "japan": "🇯🇵", "jordan": "🇯🇴", "mexico": "🇲🇽", "morocco": "🇲🇦",
    "netherlands": "🇳🇱", "new zealand": "🇳🇿", "norway": "🇳🇴", "panama": "🇵🇦",
    "paraguay": "🇵🇾", "portugal": "🇵🇹", "qatar": "🇶🇦", "saudi arabia": "🇸🇦",
    "scotland": "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "senegal": "🇸🇳", "south africa": "🇿🇦", "south korea": "🇰🇷", "spain": "🇪🇸",
    "sweden": "🇸🇪", "switzerland": "🇨🇭", "tunisia": "🇹🇳", "turkey": "🇹🇷",
    "türkiye": "🇹🇷", "usa": "🇺🇸", "uruguay": "🇺🇾", "uzbekistan": "🇺🇿",
}


def flag(name: str) -> str:
    """Flag emoji for a team, or '' for not-yet-decided knockout placeholders
    ('Winner Group A', 'Runner-up Group B', ...)."""
    return _FLAGS.get(str(name).strip().lower(), "")


# --------------------------------------------------------------------------- #
# Shared CSS (inject once per tab)
# --------------------------------------------------------------------------- #
CSS = """
<style>
.wc-card{border:1px solid #E9ECEF;border-radius:16px;padding:18px 22px;
  margin:0 0 16px 0;box-shadow:0 6px 20px rgba(30,58,95,.06);background:#fff;}
.wc-card .top{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;flex-wrap:wrap;}
.wc-teams{font-size:1.5rem;font-weight:800;color:#1C1919;line-height:1.25;}
.wc-vs{display:inline-block;background:#F59E0B;color:#fff;font-size:.8rem;
  font-weight:800;border-radius:7px;padding:1px 7px;margin:0 8px;vertical-align:middle;}
.wc-sub{color:#6B7280;font-size:.9rem;margin-top:4px;}
.wc-right{min-width:320px;flex:1;}
.wc-pick{font-size:.95rem;color:#1C1919;margin-bottom:4px;}
.wc-pick b{color:#1E3A5F;}
.wc-score{display:inline-block;background:#1E3A5F;color:#fff;font-weight:800;
  border-radius:8px;padding:2px 10px;font-size:.95rem;margin-left:6px;}
/* status badge top-right of a fixture */
.wc-status{display:inline-block;border-radius:999px;padding:2px 10px;font-size:.78rem;
  font-weight:800;vertical-align:middle;margin-left:8px;}
.wc-status.ft{background:#1E3A5F;color:#fff;font-size:1.15rem;padding:3px 13px;}
.wc-status.ft .ftlab{opacity:.7;font-size:.7rem;font-weight:700;margin-right:3px;vertical-align:middle;}
.wc-status.up{background:#FEF3C7;color:#7C2D12;border:1px solid #F59E0B;}
.wc-status.live{background:#DC2626;color:#fff;font-size:1.15rem;padding:3px 13px;
  box-shadow:0 0 0 0 rgba(220,38,38,.55);animation:wclivepulse 1.4s ease-out infinite;}
.wc-status.live .lvlab{font-size:.7rem;font-weight:800;letter-spacing:.04em;
  margin-right:5px;vertical-align:middle;}
.wc-status.live .lvdot{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:#fff;margin-right:5px;vertical-align:middle;
  animation:wclivedot 1.1s ease-in-out infinite;}
@keyframes wclivepulse{0%{box-shadow:0 0 0 0 rgba(220,38,38,.55)}
  70%{box-shadow:0 0 0 8px rgba(220,38,38,0)}100%{box-shadow:0 0 0 0 rgba(220,38,38,0)}}
@keyframes wclivedot{0%,100%{opacity:1}50%{opacity:.25}}
/* small section labels inside a card */
.wc-eyebrow{text-transform:uppercase;letter-spacing:.04em;font-size:.72rem;
  font-weight:800;color:#9CA3AF;margin:0 0 4px 0;}
.wc-chips-label{margin-top:14px;}
.wc-bar{display:flex;height:30px;border-radius:8px;overflow:hidden;
  font-size:.8rem;font-weight:700;color:#fff;}
.wc-bar .seg{display:flex;align-items:center;justify-content:center;min-width:34px;}
/* home/draw/away = three outcomes the pool picked, NOT good/bad — use a
   neutral categorical palette so a 100% segment reads as "unanimous", not "wrong". */
.wc-h{background:#1E3A5F;} .wc-d{background:#64748B;} .wc-a{background:#0D9488;}
.wc-legend{color:#6B7280;font-size:.82rem;margin-top:6px;}
.wc-chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:7px;}
.wc-chip{font-size:.85rem;border-radius:999px;padding:3px 11px;border:1px solid;
  white-space:nowrap;}
.wc-chip b{font-weight:800;}
.chip-x{background:#DCFCE7;border-color:#16A34A;color:#14532D;}
.chip-o{background:#FEF3C7;border-color:#F59E0B;color:#7C2D12;}
.chip-m{background:#F3F4F6;border-color:#D1D5DB;color:#6B7280;}
.chip-n{background:#EEF2F8;border-color:#C7D2E3;color:#1E3A5F;}

.oc-row{display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 8px 0;}
.oc-card{flex:1;min-width:150px;border:1px solid #EDEFF2;border-radius:14px;
  padding:14px 18px;background:#fff;box-shadow:0 4px 14px rgba(30,58,95,.05);}
.oc-label{color:#6B7280;font-size:.92rem;margin-bottom:6px;}
.oc-value{font-size:2.2rem;font-weight:800;color:#1C1919;line-height:1;}
.oc-delta{margin-top:8px;font-size:.92rem;font-weight:700;}
.oc-delta .oc-sub{color:#9CA3AF;font-weight:400;font-size:.82rem;}
.oc-good{color:#16A34A;} .oc-bad{color:#DC2626;} .oc-flat{color:#9CA3AF;}

/* gameweek scoreboard (points each player earned this matchday) */
.gw{width:100%;border-collapse:separate;border-spacing:0 6px;margin:2px 0 6px 0;}
.gw th{text-align:left;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;
  color:#9CA3AF;font-weight:800;padding:0 14px 2px;}
.gw td{background:#fff;padding:9px 14px;border-top:1px solid #EEF0F3;border-bottom:1px solid #EEF0F3;}
.gw td:first-child{border-left:1px solid #EEF0F3;border-radius:10px 0 0 10px;
  font-weight:700;color:#1C1919;}
.gw td:last-child{border-right:1px solid #EEF0F3;border-radius:0 10px 10px 0;text-align:right;}
.gw .rk{color:#9CA3AF;font-weight:800;margin-right:10px;}
.gw .pts{display:inline-block;background:#1E3A5F;color:#fff;font-weight:800;
  border-radius:8px;padding:2px 11px;min-width:34px;text-align:center;}
.gw .pts.zero{background:#F3F4F6;color:#9CA3AF;}
</style>
"""


# --------------------------------------------------------------------------- #
# Fixture card
# --------------------------------------------------------------------------- #
def pick_split(rows: pd.DataFrame) -> dict:
    """Distribution of the friends' predicted outcomes for one match."""
    pred = rows[rows["has_prediction"]]
    ph, pa = pred["pred_home"], pred["pred_away"]
    home, draw, away = int((ph > pa).sum()), int((ph == pa).sum()), int((pa > ph).sum())
    total = home + draw + away
    modal = pred["pred_score"].mode().iloc[0] if len(pred) else "-"
    counts = {"home": home, "draw": draw, "away": away}
    consensus = max(counts, key=counts.get) if total else None
    return {"home": home, "draw": draw, "away": away, "total": total,
            "modal": modal, "consensus": consensus,
            "blank": int((~rows["has_prediction"]).sum())}


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def match_card_html(rows: pd.DataFrame) -> str:
    """Render one match (all players) as a reference-style fixture card.

    ┌──────────────────────────────────────────────────────────────────┐
    │ ✏️ COPY in this function — all user-facing wording for a fixture   │
    │ box lives below, each tagged with a "✏️ COPY —" marker:            │
    │   (1) sub .............. the stage · date · time line              │
    │   (2) cons_txt ......... the "Consensus: …" phrasing               │
    │   (3) legend ........... the home/draw/away tally line             │
    │   (4) status .......... the FT-score / kick-off-time badge         │
    │   (4b) "Fancied:" ..... label before the pool's majority pick      │
    │   (5) chips_label ...... the legend under the prediction chips     │
    │ Layout/markup is the final f-string — leave the HTML tags intact.  │
    └──────────────────────────────────────────────────────────────────┘
    """
    meta = rows.iloc[0]
    home, away = str(meta["home"]), str(meta["away"])
    played = bool(meta["played"])
    live = bool(meta["live"]) if "live" in meta.index else False
    dt = pd.to_datetime(meta["datetime"])
    # ✏️ COPY (1) — sub-heading line under the team names.
    sub = f"{_html.escape(str(meta['stage_raw']))} · {dt:%-d %b} · {dt:%H:%M}"

    s = pick_split(rows)
    # ✏️ COPY (2) — how the pool's majority pick is described.
    cons_txt = {"home": f"{home} win", "draw": "Draw", "away": f"{away} win",
                None: "No picks yet"}[s["consensus"]]

    # distribution bar (only non-zero segments) — the coloured % bar.
    segs = []
    for key, cls, n in (("home", "wc-h", s["home"]), ("draw", "wc-d", s["draw"]),
                        ("away", "wc-a", s["away"])):
        if n:
            segs.append(f"<div class='seg {cls}' style='flex-grow:{n}'>"
                        f"{_pct(n, s['total']):.0f}%</div>")
    bar = f"<div class='wc-bar'>{''.join(segs)}</div>" if segs else ""
    # ✏️ COPY (3) — tally line beneath the bar.
    legend = (f"<div class='wc-legend'>{home} win {s['home']} · Draw {s['draw']} · "
              f"{away} win {s['away']}"
              + (f" · {s['blank']} no pick" if s["blank"] else "")
              + f" &nbsp;·&nbsp; most-picked score <b>{_html.escape(str(s['modal']))}</b></div>")

    # ✏️ COPY (4) — status badge (top-right). No panel title — the badge
    #   alone signals the match state, and the score is the hero.
    #   live -> "🔴 LIVE 2–0"   finished -> "FT 2–1"   upcoming -> "⏰ <kickoff>"
    if live:
        status = (f"<span class='wc-status live'><span class='lvdot'></span>"
                  f"<span class='lvlab'>LIVE</span>"
                  f"{_html.escape(str(meta['actual_score']))}</span>")
    elif played:
        status = (f"<span class='wc-status ft'><span class='ftlab'>FT</span>"
                  f"{_html.escape(str(meta['actual_score']))}</span>")
    else:
        status = f"<span class='wc-status up'>⏰ {dt:%-d %b · %H:%M}</span>"

    # Right panel: what everyone predicted (no title — badge says the state).
    # ✏️ COPY (4b) — "Fancied:" label before the majority pick.
    right = (f"<div class='wc-pick'>Fancied: <b>{_html.escape(cons_txt)}</b></div>"
             f"{bar}{legend}")

    # per-player chips: each friend's predicted score, best first.
    chips = []
    nailed = 0
    outcomes = 0
    order = rows.sort_values(["points", "player"], ascending=[False, True])
    for _, r in order.iterrows():
        pred = str(r["pred_score"])
        if not played:
            cls = "chip-n"
        elif r["exact_pt"] == 3:
            cls, nailed = "chip-x", nailed + 1
        elif r["outcome_pt"] == 1:
            cls, outcomes = "chip-o", outcomes + 1
        else:
            cls = "chip-m"
        chips.append(f"<span class='wc-chip {cls}'>{_html.escape(str(r['player']))} "
                     f"<b>{_html.escape(pred)}</b></span>")

    # ✏️ COPY (5) — legend under the chips (the colour key + "nailed it").
    if played:
        tally = []
        if nailed:
            tally.append(f"🎯 {nailed} dead right")
        if outcomes:
            tally.append(f"🥈 {outcomes} Correct Outcome")
        # Nobody got the exact score OR even the outcome — name and shame.
        if not tally:
            n = len(order)
            tally.append(f"😴 None right — {n} Useless boy{'s' if n != 1 else ''}")
        chips_label = ("Predictions — "
                       "<span class='chip-x' style='border-radius:4px;padding:0 5px'>exact</span> "
                       "<span class='chip-o' style='border-radius:4px;padding:0 5px'>Correct Outcome</span> "
                       "<span class='chip-m' style='border-radius:4px;padding:0 5px'>miss</span>"
                       + (f" &nbsp;·&nbsp; {' · '.join(tally)}" if tally else ""))
    else:
        chips_label = "Predictions"
    chips_html = (f"<div class='wc-legend wc-chips-label'>{chips_label}</div>"
                  f"<div class='wc-chips'>{''.join(chips)}</div>")

    return f"""
<div class="wc-card">
  <div class="top">
    <div>
      <div class="wc-teams">{flag(home)} {_html.escape(home)}
        <span class="wc-vs">VS</span>{_html.escape(away)} {flag(away)}{status}</div>
      <div class="wc-sub">{sub}</div>
    </div>
    <div class="wc-right">
      {right}
    </div>
  </div>
  {chips_html}
</div>
"""


# --------------------------------------------------------------------------- #
# Gameweek scoreboard
# --------------------------------------------------------------------------- #
def gameweek_table_html(points_by_player: dict) -> str:
    """One row per player: Name + points earned this matchday, best first.

    `points_by_player`: {player_name: points_this_matchday}. Pass a plain dict
    (e.g. Series.to_dict()). Render only when at least one match has been
    played — caller decides that.
    """
    # ✏️ COPY — column headers for the gameweek scoreboard.
    head_name, head_pts = "Name", "Points this matchday"

    items = sorted(points_by_player.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    rows = []
    for rank, (name, pts) in enumerate(items, start=1):
        pts = int(pts)
        cls = "pts zero" if pts == 0 else "pts"
        rows.append(
            f"<tr><td><span class='rk'>{rank}</span>{_html.escape(str(name))}</td>"
            f"<td><span class='{cls}'>+{pts}</span></td></tr>")
    return (f"<table class='gw'><thead><tr><th>{_html.escape(head_name)}</th>"
            f"<th style='text-align:right'>{_html.escape(head_pts)}</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


# --------------------------------------------------------------------------- #
# Tidy stat cards (Oracle style)
# --------------------------------------------------------------------------- #
def stat_cards_html(items: list[dict]) -> str:
    """items: [{label, value, delta, tone}] where tone in {good,bad,flat,None}."""
    cards = []
    for it in items:
        delta = ""
        if it.get("delta"):
            tone = it.get("tone", "flat")
            delta = (f"<div class='oc-delta oc-{tone}'>{it['delta']} "
                     f"<span class='oc-sub'>vs field</span></div>")
        cards.append(
            f"<div class='oc-card'><div class='oc-label'>{_html.escape(str(it['label']))}</div>"
            f"<div class='oc-value'>{_html.escape(str(it['value']))}</div>{delta}</div>")
    return f"<div class='oc-row'>{''.join(cards)}</div>"


def player_stat_items(stats: pd.DataFrame, who: str, rank: int) -> list[dict]:
    """Build the five tidy cards for a player vs the field median."""
    med = stats.median(numeric_only=True)
    n = len(stats)
    med_rank = (n + 1) / 2

    def fmt(x: float) -> str:
        return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"

    def card(label, val, med_val, higher_better=True, value_str=None):
        d = val - med_val
        if abs(d) < 1e-9:
            tone, delta = "flat", "— vs field"
            return {"label": label, "value": value_str or fmt(val), "delta": "—", "tone": "flat"}
        good = (d > 0) == higher_better
        arrow = "↑" if d > 0 else "↓"
        return {"label": label, "value": value_str or fmt(val),
                "delta": f"{arrow} {fmt(abs(d))}", "tone": "good" if good else "bad"}

    rank_d = med_rank - rank  # +ve = better than the middle of the pack
    rank_card = {"label": "Current rank", "value": f"#{rank}",
                 "delta": ("—" if abs(rank_d) < 1e-9
                           else f"{'↑' if rank_d > 0 else '↓'} {fmt(abs(rank_d))}"),
                 "tone": ("flat" if abs(rank_d) < 1e-9 else "good" if rank_d > 0 else "bad")}

    return [
        card("Total points", stats.loc[who, "points"], med["points"], True),
        rank_card,
        card("🎯 Exact scores", stats.loc[who, "exact"], med["exact"], True),
        card("✅ Outcome only", stats.loc[who, "outcome"], med["outcome"], True),
        card("❌ Misses", stats.loc[who, "miss"], med["miss"], higher_better=False),
    ]
