"""Fixture-card HTML for the WC 2026 Pool dashboard.

match_card_html() turns one match's long-frame rows into a card (flags, teams,
a categorical home/draw/away split of the picks, and per-player pick chips).
Returns an HTML string for st.markdown(unsafe_allow_html=True); no Streamlit
import here, so it stays unit-testable.
"""
from __future__ import annotations

import html as _html

import pandas as pd

# Flags (emoji render fine in the browser).
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
    """Flag emoji for a team, or '' for undecided knockout placeholders."""
    return _FLAGS.get(str(name).strip().lower(), "")


# Shared CSS (inject once per tab).
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
/* tiny colour swatch in the tally line — pair with .wc-h/.wc-d/.wc-a for the fill */
.wc-sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:4px;
  vertical-align:baseline;}
.wc-chips{margin-top:14px;display:flex;flex-wrap:wrap;gap:7px;}
.wc-chip{font-size:.85rem;border-radius:999px;padding:3px 11px;border:1px solid;
  white-space:nowrap;}
.wc-chip b{font-weight:800;}
.chip-x{background:#DCFCE7;border-color:#16A34A;color:#14532D;}
.chip-o{background:#FEF3C7;border-color:#F59E0B;color:#7C2D12;}
.chip-m{background:#F3F4F6;border-color:#D1D5DB;color:#6B7280;}
.chip-n{background:#EEF2F8;border-color:#C7D2E3;color:#1E3A5F;}
</style>
"""


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
    """Render one match (all players) as a fixture card."""
    meta = rows.iloc[0]
    home, away = str(meta["home"]), str(meta["away"])
    played = bool(meta["played"])
    live = bool(meta["live"]) if "live" in meta.index else False
    dt = pd.to_datetime(meta["datetime"])
    sub = f"{_html.escape(str(meta['stage_raw']))} · {dt:%-d %b} · {dt:%H:%M}"

    s = pick_split(rows)
    cons_txt = {"home": f"{home} win", "draw": "Draw", "away": f"{away} win",
                None: "No picks yet"}[s["consensus"]]

    # Distribution bar — one coloured segment per non-zero outcome.
    segs = []
    for key, cls, n in (("home", "wc-h", s["home"]), ("draw", "wc-d", s["draw"]),
                        ("away", "wc-a", s["away"])):
        if n:
            segs.append(f"<div class='seg {cls}' style='flex-grow:{n}'>"
                        f"{_pct(n, s['total']):.0f}%</div>")
    bar = f"<div class='wc-bar'>{''.join(segs)}</div>" if segs else ""
    legend = (f"<div class='wc-legend'>"
              f"<span class='wc-sw wc-h'></span>{home} win {s['home']} · "
              f"<span class='wc-sw wc-d'></span>Draw {s['draw']} · "
              f"<span class='wc-sw wc-a'></span>{away} win {s['away']}"
              + (f" · {s['blank']} no pick" if s["blank"] else "")
              + f" &nbsp;·&nbsp; most-picked score <b>{_html.escape(str(s['modal']))}</b></div>")

    # Status badge (top-right): LIVE score / FT score / kickoff time.
    if live:
        status = (f"<span class='wc-status live'><span class='lvdot'></span>"
                  f"<span class='lvlab'>LIVE</span>"
                  f"{_html.escape(str(meta['actual_score']))}</span>")
    elif played:
        status = (f"<span class='wc-status ft'><span class='ftlab'>FT</span>"
                  f"{_html.escape(str(meta['actual_score']))}</span>")
    else:
        status = f"<span class='wc-status up'>⏰ {dt:%-d %b · %H:%M}</span>"

    right = (f"<div class='wc-pick'>Fancied: <b>{_html.escape(cons_txt)}</b></div>"
             f"{bar}{legend}")

    # Per-player chips: each friend's predicted score, best first.
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

    # Legend under the chips: colour key + the day's "nailed it" tally.
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
