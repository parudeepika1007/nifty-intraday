"""Local web dashboard for the live #2 (basis) and #3 (net delta) signals.

READ ONLY. Polls Dhan snapshots on a background thread, computes both
signals + a combined directional read, and serves a self-explanatory
single-page UI. No orders are placed and nothing here is validated — the
banner in the page says so. Open http://localhost:8777 after starting.

    python -m src.live.dashboard
    python -m src.live.dashboard --port 8800 --interval 12

Uses only the Python standard library for the server (no extra deps).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

import pandas as pd

from .feed import marketfeed_ltp
from .monitor import (NIFTY_SCRIP, NIFTY_SEG, nearest_expiry,
                      net_option_delta, resolve_front_future, seed_basis)

IST = ZoneInfo("Asia/Kolkata")
DSHIFT = 1.5  # mn — "sharp" net-delta move (heuristic, UNVALIDATED)

STATE: dict = {"latest": None, "history": deque(maxlen=60), "meta": {}}
LOCK = threading.Lock()


# ---- signal interpretation (all hypotheses, see CLAUDE.md §4) --------------

def basis_label(z: float) -> tuple[str, str]:
    if z != z:  # nan
        return "warming up", "neutral"
    if z >= 2:
        return "premium RICH (≥+2σ)", "hot"
    if z <= -2:
        return "premium CHEAP (≤−2σ)", "cold"
    if abs(z) < 0.5:
        return "normal", "neutral"
    return ("mildly rich" if z > 0 else "mildly cheap",
            "warm" if z > 0 else "cool")


def delta_label(dnet: float) -> tuple[str, str]:
    if dnet > DSHIFT:
        return "bullish repositioning", "hot"
    if dnet < -DSHIFT:
        return "bearish repositioning", "cold"
    return "stable", "neutral"


def combined_read(z: float, prev_z, dnet: float) -> tuple[str, str, str]:
    """(verdict, css_class, plain-English reason)."""
    if z != z:
        return ("WARMING UP", "neutral", "Building the rolling basis window.")
    crossed_up = prev_z is not None and prev_z == prev_z and prev_z < 0 <= z
    crossed_dn = prev_z is not None and prev_z == prev_z and prev_z >= 0 > z
    if crossed_up and dnet > DSHIFT:
        return ("LONG LEAN", "long",
                "Basis-z flipped positive AND net delta shifted bullish — "
                "futures and option positioning are repositioning up together.")
    if crossed_dn and dnet < -DSHIFT:
        return ("SHORT LEAN", "short",
                "Basis-z flipped negative AND net delta shifted bearish — "
                "both venues repositioning down together.")
    if abs(z) >= 1 and ((z > 0 and dnet < -DSHIFT) or (z < 0 and dnet > DSHIFT)):
        return ("CONFLICT — STAND ASIDE", "warn",
                "Basis and option positioning disagree on direction.")
    if abs(z) >= 2 and abs(dnet) < DSHIFT:
        return ("PRICED — NO FRESH SIGNAL", "neutral",
                "Basis is stretched but not moving and net delta is stable — "
                "the move is likely already priced in.")
    return ("NO SIGNAL — WATCHING", "neutral",
            "No confluence between the two signals right now.")


# ---- polling loop ----------------------------------------------------------

def poll_loop(interval: int, zwin: int) -> None:
    front = resolve_front_future()
    expiry = nearest_expiry()
    basis_win = seed_basis(front, zwin)
    with LOCK:
        STATE["meta"] = {
            "front_symbol": front.symbol, "front_id": front.security_id,
            "opt_expiry": expiry, "zwin": zwin, "seeded": len(basis_win),
            "dshift": DSHIFT,
        }
    segs = {NIFTY_SEG: [NIFTY_SCRIP], "NSE_FNO": [front.security_id]}
    prev_net = None
    prev_z = None

    while True:
        now = dt.datetime.now(IST)
        market_open = (dt.time(9, 15) <= now.time() <= dt.time(15, 30)
                       and now.weekday() < 5)
        try:
            ltp = marketfeed_ltp(segs)
            spot = ltp[NIFTY_SEG][str(NIFTY_SCRIP)]["last_price"]
            fut = ltp["NSE_FNO"][str(front.security_id)]["last_price"]
            basis = fut - spot
            basis_win.append(basis)
            s = pd.Series(basis_win)
            z = ((basis - s.mean()) / s.std()) if len(s) >= 3 and s.std() else float("nan")

            net, ce_oi, pe_oi = net_option_delta(expiry)
            dnet = (net - prev_net) / 1e6 if prev_net is not None else 0.0
            prev_net = net
            pcr = (pe_oi / ce_oi) if ce_oi else float("nan")

            blab, bcls = basis_label(z)
            dlab, dcls = delta_label(dnet)
            verdict, vcls, reason = combined_read(z, prev_z, dnet)
            prev_z = z

            tick = {
                "time": now.strftime("%H:%M:%S"),
                "market_open": market_open,
                "spot": round(spot, 2), "fut": round(fut, 2),
                "basis": round(basis, 2),
                "basis_z": None if z != z else round(z, 2),
                "basis_label": blab, "basis_class": bcls,
                "net_delta_mn": round(net / 1e6, 2),
                "dnet_mn": round(dnet, 2),
                "delta_label": dlab, "delta_class": dcls,
                "pcr": None if pcr != pcr else round(pcr, 2),
                "verdict": verdict, "verdict_class": vcls, "reason": reason,
            }
            with LOCK:
                STATE["latest"] = tick
                STATE["history"].appendleft(tick)
        except Exception as e:
            with LOCK:
                STATE["error"] = f"{now.strftime('%H:%M:%S')} {e}"
        time.sleep(interval)


# ---- HTTP server -----------------------------------------------------------

PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>NIFTY Intraday — Live Signals</title><style>
:root{--bg:#0e1117;--card:#171c26;--bd:#262d3a;--tx:#e6edf3;--mut:#8b95a5;
--hot:#ff5c5c;--cold:#3fb950;--warm:#f0a35e;--cool:#79c0ff;--neutral:#8b95a5;
--long:#3fb950;--short:#ff5c5c;--warn:#f0a35e}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--tx)}
.wrap{max-width:1040px;margin:0 auto;padding:18px}
.banner{background:#3a2a12;border:1px solid #6b4a1a;color:#f0c987;border-radius:8px;
padding:8px 12px;font-size:13px;margin-bottom:14px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mut);font-size:13px;margin-bottom:14px}
.status{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:16px;font-size:13px;color:var(--mut)}
.status b{color:var(--tx)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
.open{background:var(--cold)}.closed{background:var(--hot)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:var(--mut);margin:0 0 10px}
.big{font-size:34px;font-weight:700;line-height:1}.unit{font-size:14px;color:var(--mut);font-weight:400}
.pill{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;margin-top:10px}
.row{display:flex;justify-content:space-between;margin-top:8px;font-size:14px}
.row span:first-child{color:var(--mut)}
.desc{color:var(--mut);font-size:12.5px;margin-top:12px;border-top:1px solid var(--bd);padding-top:10px}
.gauge{height:6px;background:#0a0d13;border-radius:4px;margin-top:12px;position:relative}
.gz{position:absolute;top:-3px;width:12px;height:12px;border-radius:50%;background:var(--tx);transform:translateX(-50%)}
.gmid{position:absolute;left:50%;top:-4px;width:1px;height:14px;background:#3a4150}
.verdict{grid-column:1/-1;text-align:center;padding:20px}
.verdict .big{font-size:30px;margin-bottom:6px}
.hot{color:var(--hot)}.cold{color:var(--cold)}.warm{color:var(--warm)}.cool{color:var(--cool)}
.neutral{color:var(--neutral)}.long{color:var(--long)}.short{color:var(--short)}.warn{color:var(--warn)}
.pill.hot{background:#3a1414;color:var(--hot)}.pill.cold{background:#10301a;color:var(--cold)}
.pill.warm{background:#3a2a12;color:var(--warm)}.pill.cool{background:#12283a;color:var(--cool)}
.pill.neutral{background:#1c222c;color:var(--neutral)}
.pill.long{background:#10301a;color:var(--long)}.pill.short{background:#3a1414;color:var(--short)}
.pill.warn{background:#3a2a12;color:var(--warn)}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:12.5px}
th,td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--bd)}
th{color:var(--mut);font-weight:500}td:first-child,th:first-child{text-align:left}
.foot{color:var(--mut);font-size:12px;margin-top:14px}
.helpbtn{font:inherit;font-size:12px;background:#1c2430;color:var(--cool);border:1px solid var(--bd);
border-radius:6px;padding:4px 10px;cursor:pointer;vertical-align:middle;margin-left:10px}
.helpbtn:hover{background:#22303f}
.info{cursor:pointer;color:var(--mut);border:1px solid var(--bd);border-radius:50%;width:17px;height:17px;
display:inline-block;text-align:center;line-height:15px;font-size:11px;font-style:italic;font-weight:700;margin-left:7px}
.info:hover{color:var(--cool);border-color:var(--cool)}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.65);display:none;z-index:50;overflow:auto}
.modal.show{display:block}
.modalbox{max-width:780px;margin:36px auto;background:var(--card);border:1px solid var(--bd);border-radius:12px}
.modalhd{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--bd);padding:15px 22px;
display:flex;justify-content:space-between;align-items:center;border-radius:12px 12px 0 0}
.modalhd h2{margin:0;font-size:16px}
.close{cursor:pointer;font-size:24px;color:var(--mut);background:none;border:0;line-height:1}
.close:hover{color:var(--tx)}
.mbody{padding:4px 22px 22px}
.metric{border-bottom:1px solid var(--bd);padding:16px 0;scroll-margin-top:64px}
.metric:last-child{border:0}.metric h3{margin:0 0 4px;font-size:14.5px}
.metric p{margin:6px 0;font-size:13px;color:#c9d3df}
.formula{font-family:ui-monospace,Consolas,monospace;background:#0a0d13;border:1px solid var(--bd);
border-radius:6px;padding:9px 11px;font-size:12.5px;color:var(--cool);margin:8px 0;white-space:pre-wrap}
.ex{font-size:12.5px;color:var(--mut)}.ex b{color:var(--tx)}
</style></head><body><div class=wrap>
<div class=banner>⚠ RESEARCH MODE — observation only, <b>no orders are placed</b>. These signals
are <b>not validated</b> (they fail / haven't met CLAUDE.md §8). Use for intuition, not trading.</div>
<h1>NIFTY Intraday — Live Signals<button id=helpbtn class=helpbtn>？ How to read this</button></h1>
<div class=sub>Idea #2 Basis Regime &nbsp;·&nbsp; Idea #3 Option Net Delta &nbsp;·&nbsp; all times IST</div>
<div class=status>
<div><span id=mk class="dot closed"></span><b id=mkt>—</b></div>
<div>Updated <b id=time>—</b></div>
<div>Spot <b id=spot>—</b></div>
<div>Front fut <b id=fut>—</b> (<span id=futsym>—</span>)</div>
<div>Opt expiry <b id=exp>—</b></div>
</div>
<div class=grid>
<div class=card><h2>#2 · Basis Regime<span class=info data-help=m-z>i</span></h2>
<div><span class=big id=basis>—</span> <span class=unit>pts (fut − spot)</span></div>
<div class=gauge><div class=gmid></div><div class=gz id=gz style="left:50%"></div></div>
<div class=row><span>basis-z (vs rolling mean)</span><b id=z>—</b></div>
<span class="pill neutral" id=bpill>—</span>
<div class=desc>How rich/cheap the futures premium is vs its own last-hour mean.
Stretched z (±2σ) flags institutional repositioning the spot hasn't caught up to.
The <b>flip through zero with momentum</b> is the directional tell, not the level itself.</div></div>
<div class=card><h2>#3 · Option Net Delta<span class=info data-help=m-delta>i</span></h2>
<div><span class=big id=nd>—</span> <span class=unit>mn (Σ δ·OI)</span></div>
<div class=row><span>change since last tick</span><b id=dnet>—</b></div>
<div class=row><span>put/call OI ratio</span><b id=pcr>—</b></div>
<span class="pill neutral" id=dpill>—</span>
<div class=desc>Aggregate delta of all open option OI on the nearest expiry. The
<b>change per tick</b> is the signal — a sharp jump = fresh call-buying / put-selling
(bullish), a sharp drop = the reverse. The absolute level is noisy.</div></div>
<div class="card verdict"><h2>Combined Directional Read (hypothesis)<span class=info data-help=m-read>i</span></h2>
<div class=big id=verdict>—</div>
<div class=desc id=reason style="border:0;color:var(--tx)">—</div></div>
</div>
<table><thead><tr><th>time</th><th>spot</th><th>fut</th><th>basis</th><th>z</th>
<th>netΔ(mn)</th><th>Δ/tick</th><th>PCR</th><th>read</th></tr></thead>
<tbody id=hist></tbody></table>
<div class=foot id=foot>connecting…</div>
</div>
<div class=modal id=modal><div class=modalbox>
<div class=modalhd><h2>How to read these signals</h2><button class=close id=closebtn>×</button></div>
<div class=mbody>
<div class=metric><h3>The idea in one line</h3>
<p>Two crowds leave footprints: <b>institutions</b> trade index <b>futures</b>, the
<b>options crowd</b> shows up in net delta. When <b>both turn the same way at the same
moment</b>, NIFTY tends to follow for the next 15–30 min. That confluence is the edge we hunt.</p></div>

<div class=metric><h3>spot &amp; fut</h3>
<p><b>spot</b> = the NIFTY index right now. <b>fut</b> = the near-month (June) futures price —
where institutions are willing to trade the index.</p></div>

<div class=metric><h3>basis</h3>
<div class=formula>basis = fut − spot</div>
<p>Futures normally trade a little <i>above</i> spot (cost of carry), so a positive basis
is normal. The raw number is <b>not</b> the signal — what matters is whether it's unusual,
which the z-score measures.</p></div>

<div class=metric id=m-z><h3>basis-z &nbsp;— the #2 signal</h3>
<div class=formula>basis_z = (basis − mean(last 60 basis)) / std(last 60 basis)</div>
<p>"Is the premium <b>weird right now</b> vs its own last hour?" The 60-min window is seeded
from today's 1-min bars so it's meaningful from the first tick.</p>
<p><b>z ≈ 0</b> → completely normal. &nbsp; <b>z ≥ +2</b> → premium unusually <span class=cold>rich</span>.
&nbsp; <b>z ≤ −2</b> → unusually <span class=cold>cheap</span>.</p>
<p class=ex>Direction comes from the <b>flip through zero</b>, not the level: z swinging
<b>negative → positive</b> = institutions just turned buyers (bullish); positive → negative = sellers (bearish).
A steadily-rich z means the move is already priced in — no edge.</p></div>

<div class=metric id=m-delta><h3>netΔ(mn) &amp; Δ/tick &nbsp;— the #3 signal</h3>
<div class=formula>netΔ = Σ over all strikes ( δ_call × OI_call + δ_put × OI_put )
       ( shown in millions; put delta is negative )
Δ/tick = netΔ(now) − netΔ(previous tick)</div>
<p><b>netΔ</b> is the aggregate directional lean of every open option position on the
nearest expiry. Dhan returns the greeks directly, so no maths assumptions on our side.</p>
<p>The <b>level is noisy — ignore it.</b> The signal is <b>Δ/tick</b>: a sharp <span class=cold>+</span> jump
= fresh call-buying / put-selling (bullish); a sharp <span class=hot>−</span> drop = the reverse.
"Sharp" currently = bigger than ±1.5 mn in one tick (a heuristic, not yet calibrated).</p></div>

<div class=metric><h3>PCR</h3>
<div class=formula>PCR = total put OI / total call OI</div>
<p>Background sentiment only. Rising = more puts (defensive/hedging), falling = more calls.
Supporting context, never a trigger on its own.</p></div>

<div class=metric id=m-read><h3>Combined Directional Read</h3>
<p>The verdict banner fires only on <b>confluence</b> of the two ⭐ signals:</p>
<div class=formula>LONG LEAN   : basis_z crosses 0 upward   AND  Δ/tick > +1.5 mn
SHORT LEAN  : basis_z crosses 0 downward AND  Δ/tick < −1.5 mn
CONFLICT    : basis &amp; options disagree (one up, one down) → stand aside
PRICED      : |z| ≥ 2 but flat, Δ/tick small → move already in, no entry
NO SIGNAL   : no confluence → do nothing (this is most of the time)</div>
<p class=ex><b>If/when it turns LONG/SHORT, the hypothesised trade:</b> take the June future
(or an at-the-money option) in that direction; stop at the recent few-minute low/high;
exit at a target <b>or</b> after 30 minutes, whichever first; risk a small fixed amount.</p>
<p class=ex style="color:var(--warn)">⚠ Not validated. The basis backtest scored below the
§8 bar and net-delta has no backtest yet. The banner is a hypothesis lighting up, not a tested
edge — don't trade real money on it until we've recorded ticks and validated forward.</p></div>
</div></div></div>
<script>
function cls(el,c){el.className=el.className.replace(/\\b(hot|cold|warm|cool|neutral|long|short|warn)\\b/g,'').trim();if(c)el.classList.add(c);}
async function tick(){
 try{const r=await fetch('/api/signals');const d=await r.json();const t=d.latest,m=d.meta;
 document.getElementById('futsym').textContent=m.front_symbol||'—';
 document.getElementById('exp').textContent=m.opt_expiry||'—';
 if(!t){document.getElementById('foot').textContent='warming up…';return;}
 const mk=document.getElementById('mk'),mkt=document.getElementById('mkt');
 mk.className='dot '+(t.market_open?'open':'closed');mkt.textContent=t.market_open?'MARKET OPEN':'MARKET CLOSED';
 document.getElementById('time').textContent=t.time;
 document.getElementById('spot').textContent=t.spot;
 document.getElementById('fut').textContent=t.fut;
 document.getElementById('basis').textContent=t.basis;
 const z=document.getElementById('z');z.textContent=t.basis_z==null?'—':t.basis_z;cls(z,t.basis_class);
 const zp=t.basis_z==null?0:Math.max(-3,Math.min(3,t.basis_z));
 document.getElementById('gz').style.left=((zp+3)/6*100)+'%';
 const bp=document.getElementById('bpill');bp.textContent=t.basis_label;bp.className='pill '+t.basis_class;
 const nd=document.getElementById('nd');nd.textContent=t.net_delta_mn;
 const dn=document.getElementById('dnet');dn.textContent=(t.dnet_mn>0?'+':'')+t.dnet_mn+' mn';cls(dn,t.delta_class);
 document.getElementById('pcr').textContent=t.pcr==null?'—':t.pcr;
 const dp=document.getElementById('dpill');dp.textContent=t.delta_label;dp.className='pill '+t.delta_class;
 const v=document.getElementById('verdict');v.textContent=t.verdict;cls(v,t.verdict_class);
 document.getElementById('reason').textContent=t.reason;
 const rows=d.history.map(h=>`<tr><td>${h.time}</td><td>${h.spot}</td><td>${h.fut}</td><td>${h.basis}</td>
 <td>${h.basis_z==null?'—':h.basis_z}</td><td>${h.net_delta_mn}</td><td>${(h.dnet_mn>0?'+':'')+h.dnet_mn}</td>
 <td>${h.pcr==null?'—':h.pcr}</td><td class=${h.verdict_class}>${h.verdict}</td></tr>`).join('');
 document.getElementById('hist').innerHTML=rows;
 document.getElementById('foot').textContent='live · '+(m.seeded||0)+' bars seeded · z-window '+m.zwin
  +' · sharp-shift threshold ±'+m.dshift+' mn'+(d.error?' · last error: '+d.error:'');
 }catch(e){document.getElementById('foot').textContent='fetch error: '+e;}
}
const modal=document.getElementById('modal');
function openModal(a){modal.classList.add('show');if(a){const el=document.getElementById(a);if(el)el.scrollIntoView({block:'start'});}}
function closeModal(){modal.classList.remove('show');}
document.getElementById('helpbtn').onclick=()=>openModal();
document.getElementById('closebtn').onclick=closeModal;
modal.addEventListener('click',e=>{if(e.target===modal)closeModal();});
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeModal();});
document.querySelectorAll('[data-help]').forEach(el=>el.onclick=()=>openModal(el.getAttribute('data-help')));
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def do_GET(self):
        if self.path.startswith("/api/signals"):
            with LOCK:
                payload = {"latest": STATE["latest"],
                           "history": list(STATE["history"]),
                           "meta": STATE["meta"],
                           "error": STATE.get("error")}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--interval", type=int, default=12, help="seconds between polls")
    ap.add_argument("--zwin", type=int, default=60)
    args = ap.parse_args()

    threading.Thread(target=poll_loop, args=(args.interval, args.zwin),
                     daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard on http://localhost:{args.port}  (Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
