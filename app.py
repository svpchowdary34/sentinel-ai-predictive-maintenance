# ==============================================================================
# SENTINEL AI — Predictive Maintenance & Equipment Reliability Intelligence | ONE Colab cell
# Detect the signal. Prevent the shutdown.
# ==============================================================================
import subprocess, sys, os, re, json, zipfile, html, math, warnings
warnings.filterwarnings("ignore")
def _deps():
    try:
        from importlib.metadata import version
        if int(version("gradio").split(".")[0]) not in (4, 5): raise RuntimeError("gradio")
        for p in ("plotly", "scipy", "scikit-learn", "requests", "pandas", "numpy"): version(p)
    except Exception:
        print("Installing dependencies (about a minute)...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scipy", "scikit-learn>=1.2", "requests", "pandas", "numpy"]); print("Dependencies installed.")
        except Exception as e: print("Install warning:", type(e).__name__, "- check internet access and re-run.")
_deps()
try:
    import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr, requests
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
except Exception as e:
    raise SystemExit("Libraries could not be imported (" + type(e).__name__ + "). Restart the runtime and re-run this cell.")

C = dict(bg="#16232B", sf="#253640", te="#80E0C0", am="#F0BF78", vi="#D2B8FF", tx="#F1F5F4", mu="#A5B6C0", co="#F18D88")
G = {}; TESTS = []; _DEMO = {}
BLANK = ["", "nan", "none", "n/a", "na", "-", "null", "nat"]
ASOF = pd.Timestamp("2026-10-26")
CH = ["temperature", "vibration", "pressure", "rotational_speed", "electrical_current", "load_percentage"]
UNIT = dict(temperature="°C", vibration="mm/s", pressure="bar", rotational_speed="rpm", electrical_current="A", load_percentage="%")
RANGE = dict(temperature=(-40, 400), vibration=(0, 100), pressure=(0, 500), rotational_speed=(0, 30000), electrical_current=(0, 2000), load_percentage=(0, 150))
DEF = dict(base_n=100, win=8, zthr=3.5, spike=6.0, persist=.6, stale=3, hz=14, labour=60.0, dt_default=24.0, w=dict(sensor=.4, trend=.15, fail=.15, service=.15, dq=.15), t_watch=80, t_insp=65, t_urg=45)
STATES = ["Normal", "Watch", "Inspection recommended", "Urgent review"]
STC = {"Normal": C["te"], "Watch": C["vi"], "Inspection recommended": C["am"], "Urgent review": C["co"], "High": C["co"], "Medium": C["am"], "Low": C["te"]}
esc = lambda x: html.escape(str(x))
norm = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
nf = lambda v: "—" if nul(v) else format(v, ",.0f")
fx = lambda v, d=1: "—" if nul(v) else format(v, "." + str(d) + "f")
pc0 = lambda v: "—" if nul(v) else format(v, ".0%")
money = lambda v: "—" if nul(v) else ("-$" if v < 0 else "$") + format(abs(v), ",.0f")
dts = lambda v: "—" if nul(v) else pd.Timestamp(v).strftime("%Y-%m-%d")
Td = lambda d: pd.Timedelta(days=float(d))
def pill(s, col=None): col = col or STC.get(s, C["mu"]); return "<span class='pill' style='color:%s;border-color:%s'>%s</span>" % (col, col, esc(s))
def note(t, k="info"):
    col = {"info": C["mu"], "ok": C["te"], "warn": C["am"], "err": C["co"], "ai": C["vi"]}[k]
    return "<div class='note' style='border-left:3px solid %s'>%s</div>" % (col, t)
def card(lab, val, desc, col=C["te"]): return "<div class='kc'><div class='kl'>%s</div><div class='kv' style='color:%s'>%s</div><div class='kd'>%s</div></div>" % (lab, col, val, desc)
H = lambda t: "<div class='sec'>%s</div>" % t
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows(): h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=380, t=""):
    f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["tx"]), height=h, title=dict(text=t, x=.02), margin=dict(l=60, r=20, t=55, b=50), legend=dict(orientation="h", y=-.25), colorway=[C["te"], C["vi"], C["am"], C["co"], C["mu"]])
    return f
def empty(msg):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(color=C["mu"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False); return sty(f, 300)
def banner(): return note("<b>Illustrative synthetic data.</b> Fictional machines, sensors and failures with planted patterns. No live telemetry and no connection to physical machines.", "warn") if G.get("demo") else ""
def clean_str(s):
    t = s.astype(str).str.strip(); return t.where(~(s.isna() | t.str.lower().isin(BLANK)), np.nan)
def to_num(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,%£€₹\s]", "", regex=True), errors="coerce")
def to_dt(s):
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def read_csv_any(f):
    p = f if isinstance(f, str) else getattr(f, "name", f)
    try: d = pd.read_csv(p)
    except Exception: d = pd.read_csv(p, sep=None, engine="python", encoding_errors="replace")
    if d.empty: raise ValueError("A file contains no rows.")
    return d
def safe(kinds):
    def deco(fn):
        def w(*a):
            def out(msg, k): return tuple((note(msg, k) if i == 0 else "") if c == "h" else empty("No data") for i, c in enumerate(kinds))
            if "X" not in G: return out("Load data first.", "info")
            try: return fn(*a)
            except Exception as e: return out("Could not render this page: " + esc(e), "err")
        return w
    return deco

# ------------------------------ FICTIONAL synthetic plant (fixed seed) ------------------------------
TYPES = [("Pump", 12, dict(temperature=62, vibration=2.2, pressure=6.5, electrical_current=38, load_percentage=70)), ("Motor", 10, dict(temperature=70, vibration=1.8, electrical_current=55, rotational_speed=1480, load_percentage=65)),
         ("Compressor", 10, dict(temperature=85, vibration=3.0, pressure=8.5, electrical_current=80, load_percentage=75)), ("Conveyor", 9, dict(vibration=1.5, electrical_current=22, rotational_speed=120, load_percentage=60)),
         ("CNC Machine", 9, dict(temperature=45, vibration=1.2, rotational_speed=8000, electrical_current=30, load_percentage=55))]
MODES = {"Pump": [("Bearing wear", "vibration", "F-BR"), ("Seal leak", "pressure", "F-SL")], "Motor": [("Bearing wear", "vibration", "F-BR"), ("Winding overheat", "temperature", "F-OH")], "Compressor": [("Overheating", "temperature", "F-OH"), ("Valve failure", "pressure", "F-VL")],
         "Conveyor": [("Bearing wear", "vibration", "F-BR"), ("Belt failure", "electrical_current", "F-BT")], "CNC Machine": [("Spindle bearing wear", "vibration", "F-SP"), ("Overheating", "temperature", "F-OH")]}
MODECH = {m[0]: m[1] for v in MODES.values() for m in v}
AMP = {"vibration": [("vibration", 1.2), ("electrical_current", .12)], "temperature": [("temperature", .35), ("vibration", .25)], "pressure": [("pressure", -.3), ("electrical_current", .1)], "electrical_current": [("electrical_current", .4)]}
FA = dict(M05=[140, 215, 290, 360], M09=[200], M11=[180], M14=[190, 330], M16=[320], M18=[250], M21=[300], M24=[210], M26=[160, 340], M29=[270], M30=[380], M35=[230], M38=[190, 370], M42=[310], M44=[350], M47=[260])
SD = dict(temperature=.02, vibration=.07, pressure=.025, rotational_speed=.004, electrical_current=.03, load_percentage=.06)
IDLE = dict(temperature=.85, vibration=.35, pressure=.6, rotational_speed=0.0, electrical_current=.4, load_percentage=.2)
PARTS = [("P01", "Bearing set", "Pump;Motor;Conveyor", 6, 14, 420, 5), ("P02", "Spindle bearing", "CNC Machine", 0, 21, 1800, 5), ("P03", "Seal kit", "Pump;Compressor", 3, 10, 260, 4), ("P04", "Valve kit", "Compressor", 1, 18, 900, 4),
         ("P05", "Cooling / thermal service kit", "Pump;Motor;Compressor;CNC Machine", 8, 7, 150, 3), ("P06", "Electrical contactor kit", "Pump;Motor;Compressor;Conveyor;CNC Machine", 4, 9, 310, 4), ("P07", "Drive belt", "Conveyor", 2, 12, 180, 3), ("P08", "Vibration sensor", "Pump;Motor;Compressor;Conveyor;CNC Machine", 5, 6, 120, 2)]
def part_for(ch, ty):
    if ch == "vibration": return "P02" if ty == "CNC Machine" else ("P04" if ty == "Compressor" else "P01")
    if ch == "temperature": return "P05"
    if ch == "pressure": return "P03"
    if ch == "electrical_current": return "P07" if ty == "Conveyor" else "P06"
    return None
def make_data(seed=11):
    r = np.random.default_rng(seed); T = 438; k = np.arange(T); tm = ASOF - pd.to_timedelta(20 * (T - 1 - k), unit="h"); mach, sen, wo, fev = [], [], [], []; n = wid = eid = 0
    crit_fix = dict(M02=5, M40=1, M05=4, M07=4, M28=5, M15=3); pm_fix = dict(M23=420, M15=150, M40=100)
    for ty, cnt, base in TYPES:
        for _ in range(cnt):
            n += 1; m = "M%02d" % n; sc = float(r.lognormal(0, .08)); mult = {c: np.ones(T) for c in base}; fl = FA.get(m, []); fm = []
            for f in fl:
                md = MODES[ty][0] if m == "M05" else MODES[ty][int(r.integers(2))]; fm.append(md)
                if m not in ("M16", "M30"):
                    g = np.clip((k - (f - 30)) / 30, 0, 1) ** 2 * (k <= f)
                    for ch, a in AMP[md[1]]:
                        if ch in mult: mult[ch] = mult[ch] + a * g
            if m == "M07": mult["vibration"] = mult["vibration"] + .6 * np.clip((k - 288) / 150, 0, 1) ** 1.5
            if m == "M12": mult["temperature"] = mult["temperature"] + .10 * np.clip((k - 330) / 108, 0, 1)
            if m == "M23": mult["vibration"] = mult["vibration"] + .5 * ((k >= 330) & (k < 420))
            if m == "M31": mult["vibration"] = mult["vibration"] + .8 * ((k >= 250) & (k <= 262))
            if m == "M15": mult["vibration"] = mult["vibration"] + .3 * np.clip((k - 380) / 57, 0, 1)
            if m == "M02": mult["temperature"] = mult["temperature"] + .06 * np.clip((k - 350) / 87, 0, 1)
            if m == "M28": mult["pressure"] = mult["pressure"] - .2 * (k >= 420); mult["electrical_current"] = mult["electrical_current"] + .5 * np.isin(k, [428, 431, 435])
            st = np.where(r.random(T) < .08, "Idle", "Running").astype(object)
            for f in fl: st[f:f + 2] = "Down"
            row = {"timestamp": tm, "machine_id": m}
            for c, b in base.items():
                v = b * sc * mult[c] * (1 + SD[c] * r.normal(size=T)); v = np.where(st != "Running", v * IDLE[c], v); v[r.random(T) < .005] = np.nan; row[c] = v
            if m == "M33": row["vibration"][-25:] = float(np.nanmedian(row["vibration"][-80:-25]))
            if m == "M45": row["temperature"][-8:] = np.nan
            row["runtime_hours"] = np.round(float(r.uniform(8000, 30000)) + np.cumsum(np.where(st == "Running", 20, 0)), 1); row["operating_state"] = st; sen.append(pd.DataFrame(row))
            mach.append(dict(machine_id=m, machine_name="%s %02d" % (ty, n), equipment_type=ty, production_line="Line " + "ABCD"[n % 4], installation_date=(ASOF - pd.Timedelta(days=int(r.integers(500, 3500)))).normalize(), criticality=crit_fix.get(m, int(r.choice([1, 2, 3, 4, 5]))),
                             operating_hours=float(row["runtime_hours"][-1]), rated_capacity=float(round(r.uniform(40, 200))), manufacturer=str(r.choice(["Vantor", "Helix", "Norrel", "Kestrel"])), maintenance_policy="deferrable (run-to-failure permitted)" if m == "M40" else ("condition-based" if n % 3 == 0 else "interval-based"), pm_interval_hours=6000.0 if m == "M40" else 2400.0))
            for f, md in zip(fl, fm):
                wid += 1; eid += 1; dh = float(r.integers(8, 38))
                fev.append(dict(event_id="FE%03d" % eid, machine_id=m, failure_timestamp=tm[f], failure_type=md[0], severity=int(r.integers(2, 5)), downtime_hours=dh, failure_confirmed=True))
                wo.append(dict(work_order_id="WO%04d" % wid, machine_id=m, maintenance_date=tm[min(f + 2, T - 1)], maintenance_type="Corrective", fault_code=md[2], failure_mode=md[0], downtime_hours=dh, repair_cost=float(round(r.lognormal(7.6, .4))), parts_cost=float(round(r.lognormal(6.5, .5))), labour_hours=round(float(r.uniform(6, 30)), 1), parts_used=part_for(md[1], ty), resolution_status="Closed", post_maintenance_observation="Sensors returned to baseline"))
            idx = pm_fix.get(m, int(r.integers(330, 431)))
            while idx > 0:
                wid += 1; wo.append(dict(work_order_id="WO%04d" % wid, machine_id=m, maintenance_date=tm[idx], maintenance_type="Preventive", fault_code=None, failure_mode=None, downtime_hours=float(r.integers(4, 9)), repair_cost=float(round(r.uniform(300, 700))), parts_cost=float(round(r.uniform(80, 400))), labour_hours=round(float(r.uniform(4, 10)), 1), parts_used=None, resolution_status="Closed", post_maintenance_observation=None))
                idx -= int(r.integers(115, 150))
    fev.append(dict(event_id="FE%03d" % (eid + 1), machine_id="M28", failure_timestamp=tm[200], failure_type="Suspected vibration alarm (not confirmed)", severity=1, downtime_hours=0.0, failure_confirmed=False))
    mach = pd.DataFrame(mach); sen = pd.concat(sen, ignore_index=True).reindex(columns=["timestamp", "machine_id"] + CH + ["runtime_hours", "operating_state"]); prod, dtw = [], []; pid = 0
    for mm in mach.itertuples():
        if mm.machine_id == "M15": segs = [(0, 9), (9, 18)]
        else:
            segs = []; d0 = int(r.integers(0, 6))
            for _ in range(int(r.integers(1, 4))): ln = int(r.integers(3, 9)); segs.append((d0, d0 + ln)); d0 += ln + int(r.integers(1, 5))
        same = mach[(mach.equipment_type == mm.equipment_type) & (mach.machine_id != mm.machine_id)].machine_id.values
        for a, b in segs:
            pid += 1; prod.append(dict(production_order_id="PO%04d" % pid, machine_id=mm.machine_id, product_id="PR%02d" % int(r.integers(1, 15)), planned_start=ASOF + Td(a), planned_end=ASOF + Td(b), priority=int(r.integers(1, 6)), required_output=float(r.integers(200, 2000)), output_per_hour=round(float(r.uniform(5, 40)), 1),
                                 machine_dependency=True if mm.machine_id == "M15" else bool(r.random() < .8), alternate_machine_id=str(r.choice(same)) if (r.random() < .3 and mm.machine_id != "M15") else None))
        for d in ([18] if mm.machine_id == "M15" else sorted(set(int(x) for x in r.integers(3, 20, int(r.integers(1, 3)))))): dtw.append(dict(machine_id=mm.machine_id, window_start=ASOF + Td(d), window_end=ASOF + Td(d + 1)))
    sk = ["Mechanical"] * 4 + ["Electrical"] * 3 + ["Instrumentation"] * 2 + ["Specialist"] + ["Mechanical", "Electrical"]
    res = pd.DataFrame([dict(technician_id="T%02d" % (i + 1), skill_category=s, available_hours=8.0, shift="Day" if i < 9 else "Night", planned_assignments=6.0 if s == "Specialist" else float(r.integers(0, 7)), location="Plant 1") for i, s in enumerate(sk)])
    parts = pd.DataFrame([dict(part_id=a, part_name=b, compatible_machine_types=c, quantity_available=d, reorder_lead_time_days=e, unit_cost=f, criticality=g) for a, b, c, d, e, f, g in PARTS])
    return dict(mach=mach, sen=sen, wo=pd.DataFrame(wo), fe=pd.DataFrame(fev), prod=pd.DataFrame(prod), res=res, parts=parts, dtw=pd.DataFrame(dtw))
def get_demo_raw():
    if "raw" not in _DEMO: _DEMO["raw"] = make_data()
    return _DEMO["raw"]
def get_demo():
    if "D" not in _DEMO: _DEMO["D"] = load_all(get_demo_raw())
    return _DEMO["D"]

# ------------------------------ ingestion and validation ------------------------------
def _sp(s):
    o = {}
    for it in s.split(","):
        p = it.strip().split(":"); o[p[0]] = (p[1], p[2] == "1", p[3].split("|") if len(p) > 3 and p[3] else [])
    return o
SPEC = dict(
 mach=_sp("machine_id:s:1:id,machine_name:s:0,equipment_type:s:1:type,production_line:s:0:line,installation_date:d:0,criticality:n:1,operating_hours:n:0,rated_capacity:n:0,manufacturer:s:0,maintenance_policy:s:0,pm_interval_hours:n:0"),
 sen=_sp("timestamp:d:1:time,machine_id:s:1,temperature:n:0,vibration:n:0,pressure:n:0,rotational_speed:n:0:rpm,electrical_current:n:0:current,load_percentage:n:0:load,runtime_hours:n:0,operating_state:s:0:state"),
 wo=_sp("work_order_id:s:1:wo_id,machine_id:s:1,maintenance_date:d:1:date,maintenance_type:s:1:type,fault_code:s:0,failure_mode:s:0,downtime_hours:n:0,repair_cost:n:0,parts_cost:n:0,labour_hours:n:0,parts_used:s:0,resolution_status:s:0,post_maintenance_observation:s:0"),
 prod=_sp("production_order_id:s:1:order_id,machine_id:s:1,product_id:s:0,planned_start:d:1:start,planned_end:d:1:end,priority:n:0,required_output:n:0,output_per_hour:n:0,machine_dependency:s:0,alternate_machine_id:s:0"),
 fe=_sp("event_id:s:1,machine_id:s:1,failure_timestamp:d:1,failure_type:s:0,severity:n:0,downtime_hours:n:0,failure_confirmed:s:0"),
 res=_sp("technician_id:s:1,skill_category:s:1:skill,available_hours:n:1,shift:s:0,planned_assignments:n:0,location:s:0"),
 parts=_sp("part_id:s:1,part_name:s:0,compatible_machine_types:s:0,quantity_available:n:1:quantity,reorder_lead_time_days:n:0,unit_cost:n:0,criticality:n:0"),
 dtw=_sp("machine_id:s:1,window_start:d:1,window_end:d:1"))
TN = dict(mach="machines", sen="sensor_readings", wo="maintenance_history", prod="production_schedule", fe="failure_events", res="maintenance_resources", parts="spare_parts", dtw="planned_downtime")
REQ = ("mach", "sen", "wo", "prod")
KEY = dict(mach=["machine_id"], sen=["machine_id", "timestamp"], wo=["work_order_id"], prod=["production_order_id"], fe=["event_id"], res=["technician_id"], parts=["part_id"], dtw=["machine_id", "window_start"])
XR = dict(prod=[("planned_end before planned_start", lambda d: d.planned_end < d.planned_start)], dtw=[("window_end before window_start", lambda d: d.window_end < d.window_start)])
def parse_map(text):
    ov = {}
    for p in re.split(r"[,;\n]", text or ""):
        if "=" in p: k, v = p.split("=", 1); ov[norm(k)] = norm(v)
    return ov
def map_cols(raw, t, ov, iss):
    d = raw.copy(); d.columns = [norm(c) for c in d.columns]; ren = {}
    for f, (k, req, al) in SPEC[t].items():
        c = ov.get(t + "_" + f) or ov.get(f); c = c if c in d.columns else next((a for a in [f] + [norm(x) for x in al] if a in d.columns and a not in ren), None)
        if c is None or c in ren:
            if req: iss.append(("error", "%s: required field '%s' not found. Columns found: %s. Use the mapping box, e.g. %s=your_column." % (TN[t], f, ", ".join(map(str, raw.columns[:12])), f)))
            continue
        if c != f and f in d.columns: d = d.drop(columns=[f])
        ren[c] = f
    d = d.rename(columns=ren); return d.loc[:, ~d.columns.duplicated()]
def rows_txt(ix): return ", ".join(str(int(i)) for i in ix[:6]) + (" (+%d more)" % (len(ix) - 6) if len(ix) > 6 else "")
def ctab(raw, t, ov, iss, ex):
    d = map_cols(raw, t, ov, iss)
    if any(l == "error" for l, _ in iss): return d
    d = d.copy(); d["_row"] = d.index + 2; bad = pd.Series(False, index=d.index)
    for f, (k, req, al) in SPEC[t].items():
        if f not in d.columns: d[f] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]") if k == "d" else np.nan
        elif k == "n": d[f] = to_num(d[f])
        elif k == "d": d[f] = to_dt(d[f])
        else: d[f] = clean_str(d[f])
    def flag(m, why):
        nonlocal bad
        m = m.fillna(False) & ~bad
        if m.any(): ex.append((TN[t], why, int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: %s (rows %s)" % (TN[t], m.sum(), why, rows_txt(d["_row"][m].values)))); bad = bad | m
    for f, (k, req, al) in SPEC[t].items():
        if req: flag(d[f].isna(), "missing " + f)
        if k == "n" and f not in CH: flag(d[f] < 0, "negative " + f)
    for why, fn in XR.get(t, []): flag(fn(d), why)
    flag(d.duplicated(KEY[t], keep="first"), "duplicate " + "+".join(KEY[t]))
    d = d[~bad].drop(columns="_row").reset_index(drop=True)
    if t == "sen":
        for c in CH:
            lo, hi = RANGE[c]; m = d[c].notna() & ((d[c] < lo) | (d[c] > hi))
            if m.any(): iss.append(("warn", "sensor_readings: %d %s values outside the plausible range %s-%s %s (check units); set to missing, not replaced." % (m.sum(), c, lo, hi, UNIT[c]))); ex.append((TN[t], c + " out of range (value set to missing)", int(m.sum()))); d.loc[m, c] = np.nan
    return d
def load_all(raw, mapping=""):
    iss, ex, ov, D = [], [], parse_map(mapping), {}; rows_in = {t: len(raw[t]) for t in SPEC if raw.get(t) is not None}
    for t in REQ:
        if raw.get(t) is None: iss.append(("error", "Missing required table: %s." % TN[t]))
    if any(l == "error" for l, _ in iss): raise ValueError(" | ".join(m for l, m in iss if l == "error"))
    for t in rows_in: D[t] = ctab(raw[t], t, ov, iss, ex)
    errs = [m for l, m in iss if l == "error"]
    if errs: raise ValueError(" | ".join(errs))
    for t in ("fe", "res", "parts", "dtw"):
        if t not in D: D[t] = pd.DataFrame({f: pd.Series(dtype="datetime64[ns]" if k == "d" else float if k == "n" else object) for f, (k, _, _) in SPEC[t].items()}); iss.append(("info", "No %s table: analyses that need it are downgraded or unavailable." % TN[t]))
    def fk(t, ids):
        m = ~D[t].machine_id.isin(ids)
        if m.any(): ex.append((TN[t], "unknown machine_id", int(m.sum()))); iss.append(("warn", "%s: %d rows excluded: unknown machine_id (not in machines)." % (TN[t], m.sum()))); D[t] = D[t][~m].reset_index(drop=True)
    ids = set(D["mach"].machine_id)
    for t in ("sen", "wo", "prod", "fe", "dtw"): fk(t, ids)
    if D["mach"].empty: raise ValueError("No valid machines remain after validation.")
    if D["sen"].empty or not D["sen"][CH].notna().any().any(): raise ValueError("No valid sensor readings remain (need at least one sensor channel with values).")
    uns = int((D["sen"].groupby("machine_id").timestamp.diff() < pd.Timedelta(0)).sum())
    if uns: iss.append(("warn", "%d sensor rows were out of timestamp order: rows were sorted (values unchanged)." % uns))
    D["sen"] = D["sen"].sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    cnt = D["sen"].groupby("machine_id").size(); short = [m for m in D["mach"].machine_id if cnt.get(m, 0) < 120]
    if short: iss.append(("warn", "%d machine(s) have fewer than 120 sensor observations: baselines and models are unreliable for them (%s)." % (len(short), ", ".join(short[:6]))))
    for c in CH:
        s = D["sen"][c]
        if s.notna().any() and s.isna().mean() > .02: iss.append(("info", "%s: %.1f%% missing (never filled in; missing values are left missing)." % (c, 100 * s.isna().mean())))
    if D["prod"].machine_dependency.isna().any(): iss.append(("info", "Some production orders have no machine_dependency flag: treated as dependent (conservative)."))
    if len(D["wo"]) < 20: iss.append(("warn", "Few maintenance records (%d): failure statistics are unreliable." % len(D["wo"])))
    D.update(issues=iss, excluded=ex, rows_in=rows_in, asof=D["sen"].timestamp.max()); return D

# ------------------------------ sensor baselines, anomalies, health ------------------------------
def robust(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 10: return np.nan, np.nan
    med = float(np.median(x)); mad = float(np.median(np.abs(x - med))) * 1.4826; return med, max(mad, .01 * abs(med), 1e-6)
def prep_S(X):
    D, P = X["D"], X["P"]; S = D["sen"].copy(); st = S.operating_state.fillna("Running").astype(str).str.lower(); S["run"] = ~st.isin(["idle", "down", "stopped", "off", "maintenance"]); base = {}
    for m, g in S.groupby("machine_id", sort=False):
        gr = g[g.run].head(P["base_n"]); base[m] = {}
        for c in CH:
            if g[c].notna().any():
                med, sc = robust(gr[c].values)
                if not np.isnan(med): base[m][c] = (med, sc)
    for c in CH:
        med = S.machine_id.map({m: v[c][0] for m, v in base.items() if c in v}).astype(float); sc = S.machine_id.map({m: v[c][1] for m, v in base.items() if c in v}).astype(float); S["z_" + c] = ((S[c] - med) / sc).where(S.run)
    S["zmax"] = S[["z_" + c for c in CH]].abs().max(axis=1); X["S"] = S.sort_values(["machine_id", "timestamp"]).reset_index(drop=True); X["base"] = base
CORR = "corrective|unplanned|breakdown|repair"
def get_fails(D):
    fe, wo = D["fe"], D["wo"]
    if len(fe):
        f = fe[fe.failure_confirmed.astype(str).str.lower().isin(["true", "1", "yes", "y"])]; return pd.DataFrame(dict(machine_id=f.machine_id.values, ts=f.failure_timestamp.values, mode=f.failure_type.values, downtime=f.downtime_hours.values)), "failure_events"
    c = wo[wo.maintenance_type.astype(str).str.lower().str.contains(CORR)]; return pd.DataFrame(dict(machine_id=c.machine_id.values, ts=c.maintenance_date.values, mode=c.failure_mode.values, downtime=c.downtime_hours.values)), "corrective work orders (no failure_events table)"
INTERP = dict(vibration="possible bearing wear, misalignment or looseness", temperature="possible cooling, lubrication or load issue", pressure="possible leak, valve or seal issue", electrical_current="possible load change or electrical issue", rotational_speed="possible drive or slip issue", load_percentage="change in operating conditions")
def analyze(X):
    D, P, S, base = X["D"], X["P"], X["S"], X["base"]; asof = X["asof"]; wo = D["wo"]; fl = X["fails"]; ev = []; rows = []; w = P["w"]
    Sr = S[S.run]; Sb = Sr.groupby("machine_id").head(P["base_n"]); mtype = D["mach"].set_index("machine_id").equipment_type; iso = {}
    for ty, gb in Sb.groupby(Sb.machine_id.map(mtype)):
        cols = [c for c in CH if gb["z_" + c].notna().mean() > .5]
        if len(gb) >= 200 and cols: iso[ty] = (cols, IsolationForest(n_estimators=100, contamination=.005, random_state=0).fit(gb[["z_" + c for c in cols]].fillna(0)))
    grp = {m: g for m, g in S.groupby("machine_id")}
    for mm in D["mach"].itertuples():
        m = mm.machine_id; g = grp.get(m)
        if g is None: continue
        gr = g[g.run]; rec = gr.tail(P["win"]); chs = list(base.get(m, {}).keys()); crit = int(mm.criticality); age = (asof - g.timestamp.max()).total_seconds() / 86400
        comp = float(g.tail(P["win"])[chs].notna().mean().mean()) if chs else 0.0; sr = tr = 0.0; faults = []; mev = []; eq = False
        def add(cat, kind, ch, z, pers, sev, interp, nxt, conf, obs=np.nan, bs=np.nan, st=None, en=None): ev.append(dict(machine_id=m, cat=cat, kind=kind, sensor=ch, start=st if st is not None else rec.timestamp.min(), end=en if en is not None else g.timestamp.max(), obs=obs, base=bs, z=z, pers=pers, sev=sev, interp=interp, nxt=nxt, conf=conf))
        for c in chs:
            med, sc = base[m][c]
            if g.tail(4)[c].isna().all(): faults.append(c); add("sensor_fault", "Possible sensor fault (dropout)", c, np.nan, np.nan, "Low", "last readings missing: wiring, power or data-link problem possible", "Check sensor calibration or wiring", "Missing values are not replaced; equipment condition for this channel is unknown"); continue
            v12 = gr.tail(12)[c].dropna()
            if len(v12) >= 8 and v12.nunique() == 1: faults.append(c); add("sensor_fault", "Possible sensor fault (flatline)", c, np.nan, np.nan, "Low", "constant reading: stuck or frozen sensor possible", "Check sensor calibration or wiring", "A flatline is excluded from equipment-condition scoring"); continue
            zr = rec["z_" + c].dropna().values
            if len(zr) < 3: continue
            zm = float(zr.mean()); mx = float(np.abs(zr).max()); pers = float((np.abs(zr) > P["zthr"]).mean()); z24 = gr.tail(24)["z_" + c].dropna().values
            slope = float(stats.linregress(np.arange(len(z24)), z24).slope) if len(z24) >= 8 else 0.0; td = float(np.sign(zm)) * slope * 10; sr = max(sr, float(np.clip((abs(zm) - 2) / (2 * P["zthr"] - 2), 0, 1)))
            if abs(zm) >= 1.5: tr = max(tr, float(np.clip(td / 3, 0, 1)))
            kind = None
            if pers >= P["persist"] and abs(zm) >= P["zthr"]: kind = "Persistent excursion"; sev = "High" if (abs(zm) >= 2 * P["zthr"] or crit >= 4) else "Medium"
            elif abs(zm) >= 2 and td >= .3: kind = "Gradual deterioration"; sev = "Medium"
            elif mx >= P["spike"] and pers < P["persist"]: kind = "Sudden spike"; sev = "Medium" if crit >= 4 else "Low"
            if kind:
                eq = True; nxt = {"Persistent excursion": "Inspect %s; compare with maintenance history; engineering review if safety-critical" % c.replace("_", " "), "Gradual deterioration": "Increase monitoring frequency and plan an inspection", "Sudden spike": "Verify the sensor and check for a process event; keep monitoring"}[kind]
                add("equipment", kind, c, zm, pers, sev, INTERP[c] + " (hypothesis, not a diagnosis)", nxt, "Recent window of %d running observations; baseline from the first %d running observations" % (len(zr), P["base_n"]), float(rec[c].mean()), med)
        if not eq and m in iso and len(rec) >= 3:
            cols, clf = iso[mm.equipment_type]; pr = clf.predict(rec[["z_" + c for c in cols]].fillna(0)); sh = float((pr == -1).mean())
            if sh >= P["persist"]: eq = True; add("equipment", "Unusual combination of sensors", "multiple", np.nan, sh, "Medium", "multi-sensor pattern unlike this equipment type's normal operation", "Inspect and compare channels together", "Isolation Forest on baseline-normalised readings; unsupervised, may include benign operating changes")
        if age > P["stale"]: faults.append("stale"); add("sensor_fault", "Stale data", "all", np.nan, np.nan, "Medium", "no recent readings: telemetry or logging gap", "Restore data feed before relying on this machine's status", "Last observation %s" % dts(g.timestamp.max()))
        w_m = wo[wo.machine_id == m]; lm = w_m[w_m.maintenance_date <= asof].maintenance_date.max() if len(w_m) else pd.NaT; hs = np.nan
        if pd.notna(lm) and g.runtime_hours.notna().any():
            rt = g.runtime_hours.ffill().bfill().values; hs = max(float(rt[-1] - np.interp(lm.value, g.timestamp.values.astype("int64"), rt)), 0.0)
        if pd.notna(lm) and (asof - lm).days <= 14 and eq:
            for e in ev[::-1]:
                if e["machine_id"] == m and e["cat"] == "equipment": e["conf"] += "; maintenance was done %s: verify work quality" % dts(lm)
        iv = mm.pm_interval_hours; ratio = hs / iv if pd.notna(hs) and pd.notna(iv) and iv > 0 else np.nan; deff = "deferrable" in str(mm.maintenance_policy).lower()
        svc = np.nan if np.isnan(ratio) else float(np.clip((ratio - .8) / .7, 0, 1)) * (.3 if deff else 1.0); n365 = int(((fl.machine_id == m) & (fl.ts > asof - Td(365))).sum()); dq = 1 - comp * (1.0 if age <= P["stale"] else 0.0)
        if faults: dq = max(dq, .6)
        f = dict(sensor=sr if chs else np.nan, trend=tr if chs else np.nan, fail=float(np.clip(n365 / 3, 0, 1)), service=svc, dq=dq); ok = [k for k in f if not np.isnan(f[k])]; sw = sum(w[k] for k in ok) or 1.0
        health = float(100 * (1 - sum(w[k] * f[k] for k in ok) / sw)); contrib = sorted([(k, 100 * w[k] * f[k] / sw) for k in ok], key=lambda t: -t[1]); NM = dict(sensor="sensor excursions", trend="deteriorating trend", fail="failure recurrence", service="service interval", dq="data quality")
        s_i = 0 if health >= P["t_watch"] else 1 if health >= P["t_insp"] else 2 if health >= P["t_urg"] else 3
        if crit >= 4 and f["sensor"] == f["sensor"] and f["sensor"] >= .8: s_i = min(3, s_i + 1)
        if n365 >= 3 or faults: s_i = max(s_i, 1)
        top = [e for e in ev if e["machine_id"] == m and e["cat"] == "equipment"]; issue = ("%s on %s" % (top[0]["kind"], top[0]["sensor"].replace("_", " "))) if top else ("sensor/data issue: " + ", ".join(faults)) if faults else ("%d failures in 12 months" % n365 if n365 >= 3 else ("service interval exceeded" if svc == svc and svc >= .5 else "none"))
        expl = "Main drivers: " + ("; ".join("%s −%.0f pts" % (NM[k], v) for k, v in contrib if v >= 1) or "none") + ". Missing: " + (", ".join(NM[k] for k in f if np.isnan(f[k])) or "nothing") + "."
        rows.append(dict(machine_id=m, name=mm.machine_name, etype=mm.equipment_type, line=mm.production_line, crit=crit, policy=mm.maintenance_policy, health=health, state=STATES[s_i], priority=(100 - health) * (.6 + .1 * crit), f_sensor=f["sensor"], f_trend=f["trend"], f_fail=f["fail"], f_service=f["service"], f_dq=f["dq"], issue=issue, expl=expl, hs=hs, interval=iv, n365=n365, age=age, comp=comp, faults=len(faults), deferrable=deff))
    mt = pd.DataFrame(rows); an = pd.DataFrame(ev, columns=["machine_id", "cat", "kind", "sensor", "start", "end", "obs", "base", "z", "pers", "sev", "interp", "nxt", "conf"]); return mt, an
def maint_stats(X):
    D, S, fl = X["D"], X["S"], X["fails"]; wo = D["wo"]; rows = []
    for mm in D["mach"].itertuples():
        m = mm.machine_id; g = S[S.machine_id == m]; w = wo[wo.machine_id == m]; f = fl[fl.machine_id == m]; rh = float(g.runtime_hours.max() - g.runtime_hours.min()) if g.runtime_hours.notna().any() else np.nan
        span = (g.timestamp.max() - g.timestamp.min()).total_seconds() / 3600 if len(g) else np.nan; cor = w[w.maintenance_type.astype(str).str.lower().str.contains(CORR)]
        rep = int(sum(((f.ts > c0) & (f.ts <= c0 + Td(60))).any() for c0 in cor.maintenance_date)); dn = cor.downtime_hours.sum()
        rows.append(dict(machine_id=m, n_fail=len(f), mtbf=rh / len(f) if len(f) and rh == rh and rh > 0 else np.nan, obs_h=rh, mttr=cor.downtime_hours.mean() if len(cor) else np.nan, down=w.downtime_hours.sum(min_count=1), cost=float((w.repair_cost.fillna(0) + w.parts_cost.fillna(0)).sum()), n_wo=len(w), n_unpl=len(cor), repeat=rep,
                         avail=1 - dn / span if span and span == span and span > 0 else np.nan, util=float(g.run.mean()) if len(g) else np.nan))
    return pd.DataFrame(rows)
def episodes(X):
    S, P, fl = X["S"], X["P"], X["fails"]; rows = []; hz = np.timedelta64(int(P["hz"] * 24), "h")
    for m, g in S[S.run].groupby("machine_id"):
        mk = (g.zmax > P["zthr"]).fillna(False).values.astype(int); d = np.diff(np.concatenate([[0], mk, [0]])); st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0] - 1; fa = np.sort(fl[fl.machine_id == m].ts.values)
        for a, b in zip(st, en):
            if b - a + 1 < 3: continue
            t1 = g.timestamp.values[b]; j = np.searchsorted(fa, t1, side="right"); rows.append(dict(machine_id=m, start=g.timestamp.values[a], end=t1, peak=float(g.zmax.values[a:b + 1].max()), followed=bool(j < len(fa) and fa[j] <= t1 + hz)))
    return pd.DataFrame(rows, columns=["machine_id", "start", "end", "peak", "followed"])

# ------------------------------ failure model (optional, labelled) ------------------------------
def model_frame(X):
    S, P, fl = X["S"], X["P"], X["fails"]; out = []; hz = np.timedelta64(int(P["hz"] * 24), "h")
    for m, g in S[S.run].groupby("machine_id"):
        g = g.sort_values("timestamp"); F = pd.DataFrame(index=g.index)
        for c in CH:
            r6 = g["z_" + c].rolling(6, min_periods=3).mean(); F["r6_" + c] = r6; F["d6_" + c] = r6 - r6.shift(6)
        F["zmaxr"] = F[["r6_" + c for c in CH]].abs().max(axis=1); t = g.timestamp.values; fa = np.sort(fl[fl.machine_id == m].ts.values)
        if len(fa): j = np.searchsorted(fa, t, side="right"); ok = j < len(fa); nx = fa[np.minimum(j, len(fa) - 1)]; y = (ok & (nx <= t + hz)).astype(int)
        else: y = np.zeros(len(t), int)
        F["y"] = y; F["machine_id"] = m; F["timestamp"] = g.timestamp.values; F["rk"] = np.arange(len(g)); out.append(F[F.rk >= P["base_n"]])
    return pd.concat(out) if out else pd.DataFrame()
def fit_model(X):
    P = X["P"]; M = model_frame(X)
    if M.empty: return dict(ok=False, why="no observations after the baseline period.")
    nfail, nm = len(X["fails"]), X["fails"].machine_id.nunique(); feats = [c for c in M.columns if c.startswith(("r6_", "d6_")) or c == "zmaxr"]
    if nfail < 8 or nm < 4 or int(M.y.sum()) < 30: return dict(ok=False, why="needs at least 8 confirmed failures on 4+ machines and 30 positive observations (found %d failures on %d machines). Showing descriptive analytics and anomaly detection only." % (nfail, nm))
    try:
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)); y = M.y.values
        oof = cross_val_predict(pipe, M[feats], y, groups=M.machine_id.values, cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]; pred = oof >= .5
        res = dict(ok=True, n=len(M), pos=int(y.sum()), prev=float(y.mean()), ap=float(average_precision_score(y, oof)), ap_base=float(average_precision_score(y, M.zmaxr.fillna(0))), prec=float(precision_score(y, pred, zero_division=0)), rec=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)), cm=confusion_matrix(y, pred), brier=float(brier_score_loss(y, oof)), meanp=float(oof.mean()))
        pipe.fit(M[feats], y); coef = pd.DataFrame(dict(feature=feats, coef=pipe[-1].coef_[0])); coef["a"] = coef.coef.abs(); res["coef"] = coef.sort_values("a", ascending=False).head(6); last = M.groupby("machine_id").tail(1); res["latest"] = dict(zip(last.machine_id, pipe.predict_proba(last[feats])[:, 1])); return res
    except Exception as e: return dict(ok=False, why="model training failed (%s)." % type(e).__name__)

# ------------------------------ production impact ------------------------------
DEPS = ["true", "1", "yes", "y", "nan"]
def prod_for(X, m, start, dur_h):
    p = X["D"]["prod"]; p = p[p.machine_id == m]; end = start + pd.Timedelta(hours=dur_h); ov = p[(p.planned_start < end) & (p.planned_end > start)]; ov = ov[ov.machine_dependency.astype(str).str.lower().isin(DEPS)]
    if ov.empty: return dict(n=0, oh=0.0, lost=0.0, alts=0, orders="")
    oh = (ov.planned_end.where(ov.planned_end < end, end) - ov.planned_start.where(ov.planned_start > start, start)).dt.total_seconds() / 3600; lost = float((oh * ov.output_per_hour).sum()) if ov.output_per_hour.notna().all() else np.nan
    return dict(n=len(ov), oh=float(oh.sum()), lost=lost, alts=int(ov.alternate_machine_id.notna().sum()), orders=", ".join(ov.production_order_id.head(4)))
def busy(X, m, d):
    p = X["D"]["prod"]; a = X["asof"].normalize() + Td(d); b = a + Td(1); p = p[(p.machine_id == m) & (p.planned_start < b) & (p.planned_end > a)]
    return len(p[p.machine_dependency.astype(str).str.lower().isin(DEPS) & p.alternate_machine_id.isna()]) > 0
def in_window(X, m, d):
    w = X["D"]["dtw"]; a = X["asof"].normalize() + Td(d); b = a + Td(1); return len(w[(w.machine_id == m) & (w.window_start < b) & (w.window_end > a)]) > 0
def px_table(X):
    D, asof, fl = X["D"], X["asof"], X["fails"]; pr = D["prod"]; rows = []; sc = X["sched"].set_index("rec_id") if len(X["sched"]) else pd.DataFrame(); rc = X["recs"].set_index("machine_id") if len(X["recs"]) else pd.DataFrame()
    for r in X["mt"][X["mt"].state != "Normal"].sort_values("priority", ascending=False).itertuples():
        p = pr[(pr.machine_id == r.machine_id) & (pr.planned_end > asof) & (pr.planned_start < asof + Td(30))]; p = p[p.machine_dependency.astype(str).str.lower().isin(DEPS)]; f = fl[fl.machine_id == r.machine_id]
        dt = float(f.downtime.mean()) if len(f) and f.downtime.notna().any() else DEF["dt_default"]; src = "mean of %d past failures" % len(f) if len(f) and f.downtime.notna().any() else "ASSUMPTION: default"; st = max(asof, p.planned_start.min()) if len(p) else asof
        pf = prod_for(X, r.machine_id, st, dt); nw = D["dtw"][(D["dtw"].machine_id == r.machine_id) & (D["dtw"].window_start >= asof)].window_start.min() if len(D["dtw"]) else pd.NaT
        sh = float(((p.planned_end.where(p.planned_end < asof + Td(30), asof + Td(30)) - p.planned_start.where(p.planned_start > asof, asof)).dt.total_seconds() / 3600).sum()) if len(p) else 0.0
        mdate = pd.NaT; mover = np.nan
        if len(rc) and r.machine_id in rc.index and len(sc):
            rid = rc.loc[r.machine_id, "rec_id"]; rid = rid if isinstance(rid, str) else rid.iloc[0]
            if rid in sc.index and sc.loc[rid, "status"] != "Unresolved": mdate = sc.loc[rid, "start"]; mover = prod_for(X, r.machine_id, mdate, sc.loc[rid, "dur"])["oh"]
        rows.append(dict(machine_id=r.machine_id, state=r.state, crit=r.crit, n_orders=len(p), sched_h=sh, req_out=float(p.required_output.sum()) if len(p) else 0.0, dt_h=dt, dt_src=src, lost=pf["lost"], alts=pf["alts"], window=nw, mdate=mdate, mover=mover, orders=pf["orders"]))
    return pd.DataFrame(rows, columns=["machine_id", "state", "crit", "n_orders", "sched_h", "req_out", "dt_h", "dt_src", "lost", "alts", "window", "mdate", "mover", "orders"])

# ------------------------------ recommendations and scheduling ------------------------------
SKILL = dict(vibration="Mechanical", temperature="Mechanical", pressure="Specialist", electrical_current="Electrical", rotational_speed="Electrical", load_percentage="Mechanical")
RC = ["rec_id", "machine_id", "issue", "evidence", "action", "state", "urgency_days", "duration_h", "skill", "part_id", "part_consumed", "cost", "priority", "assumptions", "approval"]
def make_recs(X):
    mt, an, D, P, fl = X["mt"], X["an"], X["D"], X["P"], X["fails"]; parts = D["parts"].set_index("part_id") if len(D["parts"]) else pd.DataFrame(); rows = []; k = 0
    for r in mt[mt.state != "Normal"].sort_values("priority", ascending=False).itertuples():
        a = an[(an.machine_id == r.machine_id) & (an.cat == "equipment")]; f = an[(an.machine_id == r.machine_id) & (an.cat == "sensor_fault")]; top = a.assign(az=a.z.abs().fillna(0)).sort_values("az", ascending=False).head(1)
        ch = top.sensor.iloc[0] if len(top) and top.sensor.iloc[0] in CH else None; pf = fl[fl.machine_id == r.machine_id]; prior = ch is not None and any(MODECH.get(str(x), "") == ch for x in pf["mode"]); pid = None; cons = False; skill = "Instrumentation"; svc = r.f_service
        cn = ch.replace("_", " ") if ch else ""
        if len(top) and r.state == "Urgent review": act, dur, urg, skill = "Escalate for qualified engineering assessment, then inspect %s" % cn, 4.0, 3, SKILL.get(ch, "Mechanical")
        elif len(top) and prior: act, dur, urg, cons, skill = "Perform condition-based maintenance (recurring %s failure mode)" % cn, 6.0, 7, True, SKILL.get(ch, "Mechanical")
        elif len(top) and r.state == "Inspection recommended": act, dur, urg, skill = "Inspect %s (alignment, wear, lubrication, cooling or fittings as applicable)" % cn, 2.0, 7, SKILL.get(ch, "Mechanical")
        elif len(top): act, dur, urg = "Increase monitoring frequency and review the trend", 1.0, 21
        elif len(f): act, dur, urg, pid = "Check sensor calibration or wiring", 2.0, 7, "P08"
        elif svc == svc and svc >= .5 and not r.deferrable: act, dur, urg, cons, skill, pid = "Schedule preventive servicing (interval exceeded)", 6.0, 14, True, "Mechanical", "P05"
        elif r.deferrable: act, dur, urg = "Defer under the maintenance policy; keep monitoring", 1.0, 30
        else: act, dur, urg = "Increase monitoring frequency (recurring failures)", 1.0, 21
        if ch and pid is None and (cons or r.state in ("Inspection recommended", "Urgent review")): pid = part_for(ch, r.etype)
        if pid is not None and len(parts) and (pid not in parts.index or r.etype not in str(parts.loc[pid].compatible_machine_types).split(";")): pid = None; cons = False
        cost = dur * P["labour"] + (float(parts.loc[pid].unit_cost) if cons and pid is not None and len(parts) and not nul(parts.loc[pid].unit_cost) else 0.0); k += 1
        ev = (("%s on %s: recent mean %s vs baseline %s %s (z=%+.1f, persistence %s). " % (top.kind.iloc[0], cn, fx(top.obs.iloc[0], 2), fx(top.base.iloc[0], 2), UNIT[ch], top.z.iloc[0], pc0(top.pers.iloc[0]))) if len(top) and ch and not nul(top.z.iloc[0]) else (top.kind.iloc[0] + ". " if len(top) else "")) + "Failures in 12 months: %d. %s" % (r.n365, r.expl)
        rows.append(dict(rec_id="R%03d" % k, machine_id=r.machine_id, issue=r.issue, evidence=ev, action=act, state=r.state, urgency_days=urg, duration_h=dur, skill=skill, part_id=pid, part_consumed=cons, cost=cost, priority=r.priority,
                         assumptions="Labour at %s/h, one technician. Parts are replaced only if the inspection confirms wear; an anomaly alone is not proof of a fault." % money(P["labour"]), approval="Pending qualified human review (safety-critical decisions stay with engineers)"))
    return pd.DataFrame(rows, columns=RC)
SC = ["rec_id", "machine_id", "action", "state", "day", "start", "dur", "tech", "status", "reason", "deadline"]
def schedule(X, R, naive=False):
    D = X["D"]; res = D["res"]; asof = X["asof"].normalize(); free = {t.technician_id: max(float(t.available_hours) - (0 if nul(t.planned_assignments) else float(t.planned_assignments)), 0.0) for t in res.itertuples()}; skl = {t.technician_id: str(t.skill_category) for t in res.itertuples()}
    pq = {p.part_id: float(p.quantity_available) for p in D["parts"].itertuples()}; pl = {p.part_id: (0 if nul(p.reorder_lead_time_days) else int(p.reorder_lead_time_days)) for p in D["parts"].itertuples()}; used = {}; out = []
    for t in R.sort_values(["urgency_days", "priority"], ascending=[True, False]).itertuples():
        techs = [i for i, s in skl.items() if s == t.skill]; day = None; tech = None; flag = ""; need = bool(t.part_consumed) and not nul(t.part_id); earliest = 1
        def rec_(day_, tech_, status, reason): out.append(dict(rec_id=t.rec_id, machine_id=t.machine_id, action=t.action, state=t.state, day=day_, start=(asof + Td(day_) + pd.Timedelta(hours=8)) if day_ else pd.NaT, dur=t.duration_h, tech=tech_, status=status, reason=reason, deadline=t.urgency_days))
        if naive:
            if not techs: rec_(None, None, "Unresolved", "no technician with skill " + t.skill); continue
            used[(techs[0], 1)] = used.get((techs[0], 1), 0.0) + t.duration_h; rec_(1, techs[0], "Scheduled (naive: day 1, first technician)", "ignores capacity, parts and production"); continue
        if need:
            if t.part_id not in pq: rec_(None, None, "Unresolved", "part %s not in the spare-parts catalogue" % t.part_id); continue
            if pq[t.part_id] < 1: earliest = max(1, pl[t.part_id]); flag = "part %s not in stock: procurement lead time %d d" % (t.part_id, earliest)
        last = max(int(t.urgency_days), earliest) + (14 if earliest > t.urgency_days else 0); keyf = (lambda d: d) if t.state == "Urgent review" else (lambda d: (0 if (in_window(X, t.machine_id, d) and d <= t.urgency_days) else 1, d))
        for allow in (False, True):
            if allow and t.state != "Urgent review": break
            for d in sorted(range(earliest, last + 1), key=keyf):
                if not allow and busy(X, t.machine_id, d): continue
                ok = [(free[i] - used.get((i, d), 0.0), i) for i in techs if free[i] - used.get((i, d), 0.0) >= t.duration_h - 1e-9]
                if ok: tech = max(ok)[1]; day = d; break
            if day is not None: break
        if day is None:
            mf = max([free[i] for i in techs], default=0.0); rec_(None, None, "Unresolved", ("no technician with skill %s" % t.skill) if not techs else "no %s technician has %.0f h free on any day up to day %d (max free %.1f h/day): split the job, add capacity or escalate" % (t.skill, t.duration_h, last, mf)); continue
        used[(tech, day)] = used.get((tech, day), 0.0) + t.duration_h
        if need: pq[t.part_id] -= 1
        bz = busy(X, t.machine_id, day); status = "Scheduled with production conflict" if bz else ("Scheduled after target deadline" if day > t.urgency_days else "Scheduled")
        why = "day %d (%s)%s; %s; technician %s (%s, %.1f h free that day)%s" % (day, dts(asof + Td(day)), " inside a planned downtime window" if in_window(X, t.machine_id, day) else "", "dependent production running: downtime must be absorbed or rerouted" if bz else "no dependent production without an alternate", tech, t.skill, free[tech] - used[(tech, day)] + t.duration_h, ("; " + flag) if flag else "")
        rec_(day, tech, status, why)
    S_ = pd.DataFrame(out, columns=SC); S_["over"] = False
    if naive and len(S_): g = S_.dropna(subset=["tech"]).groupby("tech").dur.sum(); S_["over"] = S_.tech.map(lambda x: bool(x in g.index and g[x] > free.get(x, 0) + 1e-9))
    return S_
def build(D, P):
    X = dict(D=D, P=P, asof=D["asof"]); prep_S(X); X["fails"], X["fsrc"] = get_fails(D); X["mt"], X["an"] = analyze(X); X["mt"] = X["mt"].merge(maint_stats(X), on="machine_id", how="left"); X["ep"] = episodes(X); X["model"] = fit_model(X)
    if X["model"]["ok"]: X["mt"]["model_risk"] = X["mt"].machine_id.map(X["model"]["latest"])
    else: X["mt"]["model_risk"] = np.nan
    X["recs"] = make_recs(X); X["sched"] = schedule(X, X["recs"]); X["naive"] = schedule(X, X["recs"], naive=True); X["px"] = px_table(X); return X

# ------------------------------ cost and downtime simulator ------------------------------
def sim_rows(q):
    p, hz = q["p"], max(q["hz"], 1); fc = q["fail_h"] * q["cph"] + q["repair"]; rows = []
    def tot(pp, wait, eff, interv, dtc):
        pw = pp * min(1.0, wait / hz); pr = (pp - pw) * (1 - eff); return interv + dtc + (pw + pr) * fc, pr, (pw + pr) * q["fail_h"] + (q["maint_h"] if dtc > 0 else 0.0)
    sc = [("Continue monitoring", lambda pp: (pp * fc, pp, pp * q["fail_h"])), ("Preventive maintenance now", lambda pp: tot(pp, 0, q["eff"], q["maint_cost"], q["maint_h"] * q["cph"] * q["share"])), ("Maintenance at next planned downtime", lambda pp: tot(pp, q["wait"], q["eff"], q["maint_cost"], 0.0)),
          ("Expedite spare part, then maintain", lambda pp: tot(pp, max(0, q["wait"] - q["saved"]), q["eff"], q["maint_cost"] + q["exped"], 0.0)), ("Defer within permitted window, then maintain", lambda pp: tot(pp, q["defer"], q["eff"], q["maint_cost"], 0.0)), ("Add a maintenance shift (maintain in 1 day)", lambda pp: tot(pp, 1, q["eff"], q["maint_cost"] + q["shift"], 0.0))]
    for nm, fn in sc:
        a, pr, dt = fn(p); lo = fn(p * .5)[0]; hi = fn(min(p * 1.5, 1.0))[0]; rows.append(dict(scenario=nm, total=a, lo=lo, hi=hi, resid=pr, dt=dt))
    return pd.DataFrame(rows)

# ------------------------------ pages ------------------------------
def flt(d, tf, lf, col_t="etype", col_l="line"):
    if tf: d = d[d[col_t].isin(tf)]
    if lf: d = d[d[col_l].isin(lf)]
    return d
@safe("hppp")
def render_fleet(tf, lf):
    X = G["X"]; mt = flt(X["mt"], tf, lf); an = X["an"][X["an"].machine_id.isin(mt.machine_id)]; eqa = an[an.cat == "equipment"]; px = X["px"][X["px"].machine_id.isin(mt.machine_id)]; sc = X["sched"]; sc = sc[sc.machine_id.isin(mt.machine_id)]
    cnt = mt.state.value_counts(); wo = X["D"]["wo"]; wo = wo[wo.machine_id.isin(mt.machine_id) & (wo.maintenance_date > X["asof"] - Td(365))]
    h = banner() + "<div class='kg'>" + card("Machines monitored", nf(len(mt)), "median health %s" % fx(mt.health.median(), 0)) + card("Urgent review", nf(cnt.get("Urgent review", 0)), "inspection recommended: %d" % cnt.get("Inspection recommended", 0), C["co"]) + card("Equipment anomalies", nf(len(eqa)), "%d high severity (recent window)" % int((eqa.sev == "High").sum()), C["am"]) + \
        card("Sensor / data issues", nf(int((an.cat == "sensor_fault").sum())), "faults, dropouts, stale data", C["vi"]) + card("Production orders exposed", nf(px.n_orders.sum()), "next 30 days, on %d non-normal machines" % len(px), C["am"]) + card("Maintenance downtime (12 mo)", nf(wo.downtime_hours.sum()) + " h", "cost %s (repair + parts)" % money((wo.repair_cost.fillna(0) + wo.parts_cost.fillna(0)).sum())) + "</div>"
    h += note("<b>Health score</b> (0-100, higher is healthier) is a transparent weighted blend of sensor excursions, trend, failure recurrence, service interval and data quality. It is <b>not a probability of failure</b>. Anomalies are <b>not confirmed failures</b>. All maintenance actions are proposals needing qualified human review.", "info")
    if len(eqa): h += H("Highest-priority alerts") + tbl(eqa.assign(o=eqa.sev.map({"High": 0, "Medium": 1, "Low": 2})).sort_values("o").head(8), [("sev", "Severity"), ("machine_id", "Machine"), ("kind", "Alert"), ("sensor", "Sensor"), ("interp", "Possible interpretation"), ("nxt", "Next step")], dict(sev=pill))
    up = sc[sc.status != "Unresolved"].sort_values("start").head(8); h += H("Upcoming proposed maintenance") + (tbl(up, [("start", "Start"), ("machine_id", "Machine"), ("action", "Action"), ("tech", "Technician"), ("status", "Status")], dict(start=dts)) if len(up) else note("Nothing scheduled.", "ok"))
    f1 = go.Figure(go.Bar(x=STATES, y=[int(cnt.get(s, 0)) for s in STATES], marker_color=[STC[s] for s in STATES])); f1 = sty(f1, 340, "Machine health states"); f1.update_yaxes(title="Machines")
    f2 = go.Figure(go.Scatter(x=mt.crit, y=100 - mt.health, mode="markers+text", text=mt.machine_id, textposition="top center", marker=dict(size=11, color=[STC[s] for s in mt.state]))); f2 = sty(f2, 380, "Equipment risk indicator vs production criticality (not a failure probability)"); f2.update_xaxes(title="Criticality (5 = highest)"); f2.update_yaxes(title="Risk indicator = 100 − health")
    t = wo.merge(X["D"]["mach"][["machine_id", "equipment_type"]], on="machine_id").groupby("equipment_type").agg(d=("downtime_hours", "sum"), c=("repair_cost", "sum")).reset_index(); f3 = go.Figure([go.Bar(x=t.equipment_type, y=t.d, name="Downtime (h)", marker_color=C["am"])]); f3 = sty(f3, 320, "Maintenance downtime by equipment type (last 12 months)"); f3.update_yaxes(title="Hours")
    return h, f1, f2, f3
@safe("hpp")
def render_health(tf, lf, q, mid):
    X = G["X"]; mt = flt(X["mt"], tf, lf)
    if q: mt = mt[mt.machine_id.str.contains(q, case=False) | mt.name.str.contains(q, case=False)]
    h = banner() + tbl(mt.sort_values("priority", ascending=False).head(60), [("machine_id", "Machine"), ("name", "Name"), ("etype", "Type"), ("crit", "Crit."), ("health", "Health"), ("state", "State"), ("model_risk", "Model risk score*"), ("issue", "Main issue"), ("expl", "Explanation"), ("avail", "Availability"), ("util", "Utilisation")], dict(health=lambda v: fx(v, 0), state=pill, model_risk=lambda v: fx(v, 2), avail=pc0, util=pc0))
    h += note("*Model risk score is a class-weighted ranking from the optional failure model (see Failure Intelligence), not a calibrated probability, and is based on synthetic labels in the demo.", "warn")
    f1 = empty("Select a machine"); f2 = empty("No maintenance history")
    if mid and mid in set(X["mt"].machine_id):
        r = X["mt"].set_index("machine_id").loc[mid]; w = X["P"]["w"]; ks = [("sensor", "f_sensor"), ("trend", "f_trend"), ("fail", "f_fail"), ("service", "f_service"), ("dq", "f_dq")]; ok = [(k, r[c]) for k, c in ks if not nul(r[c])]; sw = sum(w[k] for k, _ in ok) or 1.0
        f1 = go.Figure(go.Bar(x=[k for k, _ in ok], y=[100 * w[k] * v / sw for k, v in ok], marker_color=C["am"])); f1 = sty(f1, 320, "Health points lost by factor: %s (health %s, %s)" % (mid, fx(r.health, 0), r.state)); f1.update_yaxes(title="Points lost")
        wo = X["D"]["wo"]; wo = wo[wo.machine_id == mid].sort_values("maintenance_date"); b = X["base"].get(mid, {}); S = X["S"]; rec = S[(S.machine_id == mid) & S.run].tail(X["P"]["win"])
        h += H("Baselines vs recent readings: " + esc(mid)) + tbl(pd.DataFrame([dict(c=c, u=UNIT[c], med=v[0], rec=rec[c].mean(), z=rec["z_" + c].mean()) for c, v in b.items()]), [("c", "Sensor"), ("u", "Unit"), ("med", "Baseline median"), ("rec", "Recent mean"), ("z", "Robust z")], dict(med=lambda v: fx(v, 2), rec=lambda v: fx(v, 2), z=lambda v: "%+.1f" % v if v == v else "—"))
        h += H("Maintenance history") + (tbl(wo.tail(12), [("maintenance_date", "Date"), ("maintenance_type", "Type"), ("fault_code", "Fault code"), ("failure_mode", "Failure mode"), ("downtime_hours", "Downtime h"), ("repair_cost", "Repair cost"), ("parts_cost", "Parts cost"), ("resolution_status", "Status")], dict(maintenance_date=dts, repair_cost=money, parts_cost=money)) if len(wo) else note("No maintenance history for this machine.", "warn"))
        if len(wo): m_ = wo.assign(m=wo.maintenance_date.dt.to_period("M").dt.to_timestamp()).groupby("m").agg(c=("repair_cost", "sum"), p=("parts_cost", "sum")).reset_index(); f2 = go.Figure([go.Bar(x=m_.m, y=m_.c, name="Repair cost", marker_color=C["co"]), go.Bar(x=m_.m, y=m_.p, name="Parts cost", marker_color=C["vi"])]); f2.update_layout(barmode="stack"); f2 = sty(f2, 320, "Maintenance cost by month: " + mid)
    return h, f1, f2
@safe("hppp")
def render_sensor(mid, ch):
    X = G["X"]; S, P = X["S"], X["P"]; mid = mid or X["mt"].machine_id.iloc[0]; ch = ch or "vibration"; h = banner(); g = S[S.machine_id == mid]
    if ch not in X["base"].get(mid, {}): return h + note("%s has no usable %s channel (missing or no baseline)." % (esc(mid), esc(ch)), "warn"), empty("No data"), empty("No data"), empty("No data")
    med, sc = X["base"][mid][ch]; run = g[g.run]; an = X["an"][(X["an"].machine_id == mid)]; hi = run[run["z_" + ch].abs() > P["zthr"]]
    f1 = go.Figure([go.Scatter(x=g.timestamp, y=g[ch], mode="lines", name="Reading", line=dict(color=C["mu"], width=1)), go.Scatter(x=run.timestamp, y=run[ch].rolling(P["win"], min_periods=3).mean(), name="Rolling mean (%d obs)" % P["win"], line=dict(color=C["te"], width=2.5)),
                    go.Scatter(x=hi.timestamp, y=hi[ch], mode="markers", name="|z| > %.1f" % P["zthr"], marker=dict(color=C["co"], size=6))]); f1.add_hrect(y0=med - P["zthr"] * sc, y1=med + P["zthr"] * sc, fillcolor="rgba(128,224,192,.10)", line_width=0)
    f1.add_hline(y=med, line_color=C["te"], line_dash="dot")
    for _, w in X["D"]["wo"][X["D"]["wo"].machine_id == mid].iterrows(): f1.add_shape(type="line", x0=w.maintenance_date, x1=w.maintenance_date, y0=0, y1=1, yref="paper", line=dict(color=C["co"] if "orrect" in str(w.maintenance_type) else C["vi"], dash="dot", width=1))
    f1 = sty(f1, 440, "%s: %s (%s). Band = baseline median ± %.1f robust σ; red dotted = corrective work, violet dotted = preventive" % (mid, ch.replace("_", " "), UNIT[ch], P["zthr"])); f1.update_yaxes(title=UNIT[ch])
    cc = run[[c for c in CH if run[c].notna().sum() > 20]].corr(); f2 = go.Figure(go.Heatmap(z=cc.values, x=list(cc.columns), y=list(cc.index), zmin=-1, zmax=1, colorscale=[[0, C["co"]], [.5, C["sf"]], [1, C["te"]]], hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>")) if len(cc) > 1 else empty("Not enough channels"); f2 = sty(f2, 340, "Sensor correlation for %s (running observations; association only)" % mid)
    dq = X["mt"].sort_values("faults", ascending=False).assign(fresh=lambda d: d.age.map(lambda a: "stale" if a > P["stale"] else "ok")); f3 = go.Figure(go.Bar(x=dq.machine_id.head(25), y=100 * dq.comp.head(25), marker_color=[C["co"] if a > P["stale"] or f > 0 else C["te"] for a, f in zip(dq.age.head(25), dq.faults.head(25))])); f3 = sty(f3, 300, "Recent data completeness by machine (coral = stale or sensor fault)"); f3.update_yaxes(title="% complete", range=[0, 105])
    h += note("Persistent excursions, gradual drift and spikes are different signals, and a sensor fault is separated from an equipment anomaly where the evidence allows. Idle and down observations are excluded.", "info")
    if len(an): h += H("Alerts for " + esc(mid)) + tbl(an, [("sev", "Severity"), ("cat", "Class"), ("kind", "Alert"), ("sensor", "Sensor"), ("start", "From"), ("obs", "Observed"), ("base", "Baseline"), ("z", "z"), ("pers", "Persistence"), ("interp", "Possible interpretation"), ("nxt", "Suggested step"), ("conf", "Confidence / limits")], dict(sev=pill, start=dts, obs=lambda v: fx(v, 2), base=lambda v: fx(v, 2), z=lambda v: "%+.1f" % v if v == v else "—", pers=pc0))
    return h, f1, f2, f3
@safe("hppp")
def render_fail(mid):
    X = G["X"]; mt = X["mt"]; fl = X["fails"]; M = X["model"]; ep = X["ep"]
    h = banner() + note("<b>Definitions.</b> Failures come from %s. MTBF = running hours observed ÷ failures (if there are no failures the value is censored and shown as '—', never as zero). MTTR = mean downtime of corrective work orders. Sensor-failure correlations are associations, not causes." % esc(X["fsrc"]), "info")
    t = mt.sort_values("n_fail", ascending=False).head(25); h += tbl(t, [("machine_id", "Machine"), ("etype", "Type"), ("n_fail", "Failures"), ("mtbf", "MTBF (h)"), ("mttr", "MTTR (h)"), ("repeat", "Repeat failures <60d"), ("n_unpl", "Unplanned WOs"), ("n_wo", "All WOs"), ("down", "Downtime h"), ("cost", "Cost")], dict(mtbf=nf, mttr=fx, down=nf, cost=money))
    if len(ep): h += note("<b>Anomaly episodes</b> (3+ consecutive running observations beyond %.1f robust σ): %d found; %d (%s) were followed by a confirmed failure within %d days. The rest were anomalies without failure: an anomaly is not an impending failure." % (X["P"]["zthr"], len(ep), int(ep.followed.sum()), pc0(ep.followed.mean()), X["P"]["hz"]), "warn")
    h += H("Failure prediction (optional, labelled)")
    if not M["ok"]: h += note("Failure model disabled: " + esc(M["why"]), "info")
    else:
        cm = M["cm"]; h += note("<b>Target:</b> a confirmed failure within the next %d days. <b>Validation:</b> machine-grouped 5-fold cross-validation (all observations of one machine stay in one fold, so a failure sequence never leaks across train and test); baseline period excluded; class-weighted logistic regression, seed 0. <b>Results (held-out)</b>: PR-AUC %.2f vs %.2f for a simple 'largest sensor deviation' baseline (prevalence %.1f%%); at score ≥ 0.5 precision %.2f, recall %.2f, F1 %.2f; confusion matrix [[TN %d, FP %d],[FN %d, TP %d]]. Brier %.3f; mean score %.2f vs observed rate %.3f, so scores are <b>not calibrated probabilities</b> (class weighting inflates them). Lower the threshold for fewer missed failures at the cost of more false alarms. %s" % (X["P"]["hz"], M["ap"], M["ap_base"], 100 * M["prev"], M["prec"], M["rec"], M["f1"], cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1], M["brier"], M["meanp"], M["prev"], "<b>Synthetic labels: these metrics demonstrate the workflow, not real-world accuracy.</b>" if G.get("demo") else "Verify on your own plant before relying on it."), "ai")
        h += tbl(M["coef"], [("feature", "Feature the model relies on"), ("coef", "Standardised coefficient")], dict(coef=lambda v: "%+.2f" % v)) + note("Feature weights describe model reliance, not causation.", "warn")
    h += note("<b>Remaining useful life: disabled.</b> RUL needs run-to-failure sequences with a defined end point and consistent operating conditions; this dataset has repaired machines, mixed types and few repeat cycles, so no RUL is invented. Condition indicators (health, trend, z-scores) are shown instead.", "warn")
    fc = mt[mt.n_fail > 0].groupby("etype").n_fail.sum().reset_index(); f1 = sty(go.Figure(go.Bar(x=fc.etype, y=fc.n_fail, marker_color=C["co"])), 320, "Confirmed failures by equipment type (12 months)") if len(fc) else empty("No failures recorded")
    f2 = empty("No fault codes"); wo = X["D"]["wo"]; wc = wo[wo.fault_code.notna()]
    if len(wc): c = wc.groupby("fault_code").agg(n=("work_order_id", "size"), d=("downtime_hours", "sum")).reset_index(); f2 = go.Figure([go.Bar(x=c.fault_code, y=c.n, name="Occurrences", marker_color=C["am"]), go.Bar(x=c.fault_code, y=c.d, name="Downtime (h)", marker_color=C["vi"])]); f2.update_layout(barmode="group"); f2 = sty(f2, 320, "Recurring fault codes: occurrences and downtime")
    f3 = empty("Select a machine"); S = X["S"]
    if mid:
        g = S[(S.machine_id == mid) & S.run]; f3 = go.Figure(go.Scatter(x=g.timestamp, y=g.zmax, mode="lines", name="Largest |z| across sensors", line=dict(color=C["te"], width=1.5)))
        for _, f in fl[fl.machine_id == mid].iterrows(): f3.add_shape(type="line", x0=f.ts, x1=f.ts, y0=0, y1=1, yref="paper", line=dict(color=C["co"], width=2))
        f3 = sty(f3, 340, "Sensor deviation vs confirmed failures (red lines) for " + mid); f3.update_yaxes(title="Robust |z|")
    return h, f1, f2, f3
@safe("hppp")
def render_plan(tf, lf):
    X = G["X"]; R = X["recs"]; Sc = X["sched"]; ids = set(flt(X["mt"], tf, lf).machine_id); R = R[R.machine_id.isin(ids)]; Sc = Sc[Sc.machine_id.isin(ids)]
    h = banner() + note("<b>Scheduler:</b> deterministic priority-ordered placement (urgency, then risk). OR-Tools is not used in this build, so the schedule may not be globally optimal. Hard constraints: a technician's free hours per day are never exceeded, the required skill must exist, consumed parts must be in stock (otherwise the procurement lead time delays the task), and a task must fit inside one day. Unplaceable tasks are reported, never forced. Nothing is executed.", "warn")
    if len(R): h += H("Recommended actions") + tbl(R, [("rec_id", "ID"), ("machine_id", "Machine"), ("state", "State"), ("issue", "Issue"), ("action", "Action"), ("urgency_days", "Target (days)"), ("duration_h", "Hours"), ("skill", "Skill"), ("part_id", "Part"), ("cost", "Est. cost"), ("evidence", "Evidence"), ("assumptions", "Assumptions"), ("approval", "Approval")], dict(state=pill, cost=money))
    else: h += note("No machine needs action.", "ok")
    nv, pv = X["naive"], Sc; nv = nv[nv.machine_id.isin(ids)]; pv_conf = int((pv.status == "Scheduled with production conflict").sum()); nconf = int(sum(busy(X, r.machine_id, 1) for r in nv[nv.day.notna()].itertuples())); nover = int(nv.over.sum())
    h += H("Baseline vs proposed schedule") + tbl(pd.DataFrame([("Tasks placed", int(nv.day.notna().sum()), int((pv.status != "Unresolved").sum())), ("Technician overbooking (tasks)", nover, 0), ("Tasks on a day with dependent production", nconf, pv_conf), ("Tasks after target deadline", 0, int((pv.status == "Scheduled after target deadline").sum())), ("Unresolved tasks", int(nv.day.isna().sum()), int((pv.status == "Unresolved").sum()))], columns=["m", "b", "p"]), [("m", "Measure"), ("b", "Baseline (naive: everything on day 1, first technician)"), ("p", "Proposed")])
    if len(pv): h += H("Proposed schedule with rationale") + tbl(pv.sort_values("start"), [("rec_id", "ID"), ("machine_id", "Machine"), ("action", "Action"), ("start", "Start"), ("dur", "Hours"), ("tech", "Technician"), ("status", "Status"), ("reason", "Why this window / conflict")], dict(start=lambda v: "—" if nul(v) else pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")))
    un = pv[pv.status == "Unresolved"]
    if len(un): h += note("<b>Unresolved constraints:</b> " + esc("; ".join("%s (%s): %s" % (r.machine_id, r.rec_id, r.reason) for r in un.itertuples())), "err")
    pt = X["D"]["parts"]; 
    if len(R) and len(pt): need = R[R.part_id.notna()].merge(pt, on="part_id", how="left"); h += H("Spare-part requirements") + tbl(need, [("rec_id", "Rec."), ("machine_id", "Machine"), ("part_id", "Part"), ("part_name", "Name"), ("part_consumed", "Consumed?"), ("quantity_available", "In stock"), ("reorder_lead_time_days", "Lead time (d)")], dict(quantity_available=nf))
    f1 = empty("Nothing scheduled"); ok = pv[pv.status != "Unresolved"]
    if len(ok): f1 = go.Figure(go.Bar(base=ok.start, x=ok.dur * 3600000, y=ok.machine_id, orientation="h", marker_color=[C["co"] if "conflict" in s else (C["am"] if "after" in s else C["te"]) for s in ok.status], text=ok.tech, hovertext=ok.action)); f1.update_xaxes(type="date"); f1 = sty(f1, 420, "Maintenance calendar (teal = clean, amber = after target, coral = production conflict)")
    f2 = go.Figure([go.Bar(x=["Baseline", "Proposed"], y=[nover, 0], name="Technician overbooking", marker_color=C["co"]), go.Bar(x=["Baseline", "Proposed"], y=[nconf, pv_conf], name="Production conflicts", marker_color=C["am"]), go.Bar(x=["Baseline", "Proposed"], y=[int(nv.day.isna().sum()), int(len(un))], name="Unresolved", marker_color=C["vi"])]); f2.update_layout(barmode="group"); f2 = sty(f2, 320, "Schedule quality: baseline vs proposed")
    f3 = empty("No data"); 
    if len(R): c = R.groupby("skill").duration_h.sum().reset_index(); f3 = sty(go.Figure(go.Bar(x=c.skill, y=c.duration_h, marker_color=C["te"])), 300, "Technician hours required by skill"); f3.update_yaxes(title="Hours")
    return h, f1, f2, f3
@safe("hp")
def render_prodimp(tf, lf):
    X = G["X"]; px = X["px"]; px = px[px.machine_id.isin(flt(X["mt"], tf, lf).machine_id)]
    h = banner() + note("<b>Four different things are kept apart.</b> Observed downtime = past work orders. Scheduled maintenance duration = proposed actions. <b>Hypothetical</b> failure downtime = this machine's mean past failure downtime (or an <b>assumed</b> %d h default if it has none). Estimated production impact = overlapping dependent hours × output rate; if a rate or dependency is missing, no loss is invented ('not estimated'). Orders with a supplied alternate machine are counted but could be rerouted." % DEF["dt_default"], "info")
    if len(px): h += tbl(px, [("machine_id", "Machine"), ("state", "State"), ("crit", "Crit."), ("n_orders", "Dependent orders (30 d)"), ("sched_h", "Scheduled h"), ("req_out", "Required output"), ("dt_h", "Hypothetical failure downtime h"), ("dt_src", "Source"), ("lost", "Output at risk if it fails"), ("alts", "Orders with alternates"), ("window", "Next planned window"), ("mdate", "Proposed maintenance"), ("mover", "Production hours lost to maintenance")],
                              dict(state=pill, sched_h=nf, req_out=nf, dt_h=nf, lost=lambda v: "not estimated" if nul(v) else nf(v), window=dts, mdate=dts, mover=lambda v: "—" if nul(v) else fx(v)))
    else: h += note("No non-normal machine has dependent production in this view.", "ok")
    h += note("<b>Act now vs defer:</b> compare 'Production hours lost to maintenance' (a planned, short, controlled stop) with 'Output at risk if it fails' (an uncertain, longer, uncontrolled stop). Maintenance proposed inside a planned downtime window or on a day without dependent production costs no output. The Cost & Scenario Lab prices the trade-off with explicit assumptions.", "ai")
    f1 = empty("Nothing exposed")
    if len(px): d = px[px.lost.notna() | (px.sched_h > 0)]; f1 = go.Figure([go.Bar(x=d.machine_id, y=d.sched_h, name="Scheduled hours (30 d)", marker_color=C["mu"]), go.Bar(x=d.machine_id, y=d.dt_h, name="Hypothetical failure downtime (h)", marker_color=C["co"])]); f1.update_layout(barmode="group"); f1 = sty(f1, 360, "Production exposure: scheduled hours vs hypothetical failure downtime"); f1.update_yaxes(title="Hours")
    return h, f1
@safe("hpp")
def render_sim(mid, p, eff, cph, fh, rep, mh, mc, wait, exped, saved, shift, defer, share):
    X = G["X"]; mid = mid or X["mt"].sort_values("priority", ascending=False).machine_id.iloc[0]; R = X["recs"]; r = R[R.machine_id == mid]; fl = X["fails"]; f = fl[fl.machine_id == mid]; wo = X["D"]["wo"]; cor = wo[(wo.machine_id == mid) & wo.maintenance_type.astype(str).str.lower().str.contains(CORR)]
    nw = X["D"]["dtw"][(X["D"]["dtw"].machine_id == mid) & (X["D"]["dtw"].window_start >= X["asof"])].window_start.min() if len(X["D"]["dtw"]) else pd.NaT
    q = dict(p=float(p or 0), eff=float(eff or 0), cph=float(cph or 0), hz=X["P"]["hz"], maint_h=float(mh) if mh else (float(r.duration_h.iloc[0]) if len(r) else 4.0), maint_cost=float(mc) if mc else (float(r.cost.iloc[0]) if len(r) else 500.0), fail_h=float(fh) if fh else (float(f.downtime.mean()) if len(f) and f.downtime.notna().any() else DEF["dt_default"]),
             repair=float(rep) if rep else (float((cor.repair_cost.fillna(0) + cor.parts_cost.fillna(0)).mean()) if len(cor) else 5000.0), wait=float(wait) if wait else (float((nw - X["asof"]).days) if pd.notna(nw) else 7.0), exped=float(exped or 0), saved=float(saved or 0), shift=float(shift or 0), defer=float(defer or 0), share=float(share if share is not None else 1))
    T = sim_rows(q); T["pct"] = T.total / max(T.total.max(), 1e-9)
    h = banner() + note("<b>Hypothetical scenarios for %s, not predictions.</b> Formulas: failure cost = failure downtime h × cost/h + unplanned repair cost; expected loss = (risk during wait + residual risk after maintenance) × failure cost, where risk during wait = p × min(1, wait ÷ %d-day horizon) and residual = (p − that) × (1 − effectiveness). <b>p = %.0f%% is YOUR assumption</b> (no validated probability exists); effectiveness %.0f%%. Inputs left at 0 use data-derived defaults: maintenance %.1f h / %s, failure downtime %.0f h, repair cost %s, wait %.0f d. Range = p × 0.5 to p × 1.5." % (esc(mid), X["P"]["hz"], 100 * q["p"], 100 * q["eff"], q["maint_h"], money(q["maint_cost"]), q["fail_h"], money(q["repair"]), q["wait"]), "warn")
    h += tbl(T, [("scenario", "Scenario"), ("total", "Expected total cost"), ("lo", "Low (p×0.5)"), ("hi", "High (p×1.5)"), ("resid", "Residual risk indicator"), ("dt", "Expected downtime h")], dict(total=money, lo=money, hi=money, resid=pc0, dt=fx)) + note("Lower expected cost does not mean guaranteed savings. 'Defer' is valid only where the maintenance policy or safety rules permit it. Cost per downtime hour of 0 gives intervention-only comparisons.", "info")
    f1 = go.Figure(go.Bar(x=T.scenario, y=T.total, marker_color=C["te"], error_y=dict(type="data", symmetric=False, array=T.hi - T.total, arrayminus=T.total - T.lo))); f1 = sty(f1, 380, "Expected total cost by scenario (bars show p × 0.5 to p × 1.5)"); f1.update_yaxes(title="Cost"); f1.update_xaxes(tickangle=-20)
    f2 = go.Figure()
    for pp in [0, .05, .1, .2, .3, .4, .5]: pass
    ps = np.linspace(0, .6, 13); rows = [sim_rows(dict(q, p=float(pp))) for pp in ps]
    for i, nm in enumerate(T.scenario): f2.add_trace(go.Scatter(x=ps, y=[rr.total.iloc[i] for rr in rows], name=nm, mode="lines"))
    f2 = sty(f2, 380, "Sensitivity: expected total cost vs assumed failure probability p"); f2.update_xaxes(title="Assumed p (within horizon)", tickformat=".0%"); f2.update_yaxes(title="Expected total cost"); return h, f1, f2
@safe("hh")
def render_dq():
    D = G["D"]; X = G["X"]; h = banner() + note("Rows read: " + ", ".join("%s %s" % (TN[k], nf(v)) for k, v in D["rows_in"].items()) + ". Raw uploads stay unchanged; a validated copy is analysed. Sensor values outside plausible ranges are set to missing (never replaced). Excluded rows: <b>%d</b> (listed; nothing is dropped silently). Expected units: °C, mm/s, bar, rpm, A, %%." % sum(c for _, _, c in D["excluded"]), "ok")
    h += "".join(note(esc(m), "err" if l == "error" else "warn" if l == "warn" else "info") for l, m in D["issues"]) or note("No validation issues.", "ok")
    if D["excluded"]: h += H("Excluded rows by reason") + tbl(pd.DataFrame(D["excluded"], columns=["t", "w", "c"]), [("t", "Table"), ("w", "Reason"), ("c", "Rows")], dict(c=nf))
    return h, H("Self-test results (run at start-up)") + "".join(note(("PASS " if ok else "FAIL ") + esc(g + ": " + n) + " — " + esc(d), "ok" if ok else "err") for g, n, ok, d in TESTS)

# ------------------------------ SENTINEL Strategist ------------------------------
def ask(q, use_llm=False):
    if "X" not in G: return "Load data first."
    X = G["X"]; mt = X["mt"]; q = (q or "").strip() or "Which machines need inspection first?"; ql = q.lower(); parts = []; ids = re.findall(r"\bM\d{2}\b", q, re.I); mid = ids[0].upper() if ids else None; R = X["recs"]
    if re.search(r"inspect|first|priorit|urgent", ql): t = mt[mt.state != "Normal"].sort_values("priority", ascending=False).head(5); parts.append(("inspection_priority", "; ".join("%s (%s, health %.0f, %s)" % (r.machine_id, r.state, r.health, r.issue) for r in t.itertuples()) or "no machine is outside Normal"))
    if mid and re.search(r"why|classif|risk|explain", ql) and mid in set(mt.machine_id): r = mt.set_index("machine_id").loc[mid]; parts.append(("machine_explanation", "%s is '%s' with health %.0f/100 (not a failure probability). Issue: %s. %s Failures in 12 months: %d." % (mid, r.state, r.health, r.issue, r.expl, r.n365)))
    if re.search(r"sensor|changed|reading", ql):
        a = X["an"][X["an"].cat == "equipment"].assign(az=lambda d: d.z.abs().fillna(0)).sort_values("az", ascending=False).head(5); parts.append(("sensor_changes", "; ".join("%s %s: %s, recent %s vs baseline %s %s (z %+.1f)" % (r.machine_id, r.sensor, r.kind, fx(r.obs, 2), fx(r.base, 2), UNIT.get(r.sensor, ""), r.z) for r in a.itertuples() if not nul(r.z)) or "no persistent sensor changes"))
    if re.search(r"recur|fault|failure", ql) and not mid: wc = X["D"]["wo"].fault_code.value_counts().head(4); parts.append(("recurring_failures", "Most frequent fault codes: " + "; ".join("%s ×%d" % (k, v) for k, v in wc.items()) + ". Failures by machine: " + "; ".join("%s ×%d" % (r.machine_id, r.n_fail) for r in mt.sort_values("n_fail", ascending=False).head(4).itertuples() if r.n_fail > 0)))
    if re.search(r"disrupt|production|next week", ql): p = X["px"].sort_values(["crit", "n_orders"], ascending=False).head(5); parts.append(("production_exposure", "; ".join("%s (%s): %d dependent orders in 30 days, failure would put %s output at risk" % (r.machine_id, r.state, r.n_orders, "an unquantified amount of" if nul(r.lost) else nf(r.lost) + " units of") for r in p.itertuples()) or "no exposed machines"))
    if re.search(r"now|defer|planned downtime|wait|schedule", ql): m_ = mid or (R.machine_id.iloc[0] if len(R) else None); s = X["sched"]; s = s[s.machine_id == m_] if m_ else s.head(0); parts.append(("timing", ("%s: proposed %s; %s. Compare costs in the Cost & Scenario Lab (hypothetical)." % (m_, dts(s.start.iloc[0]) if len(s) and not nul(s.start.iloc[0]) else "not schedulable", s.reason.iloc[0] if len(s) else "no action needed")) if m_ else "no machine needs action"))
    if re.search(r"part|technician", ql): parts.append(("resources", "; ".join("%s: %s h %s%s" % (r.machine_id, r.duration_h, r.skill, (", part " + r.part_id) if not nul(r.part_id) else "") for r in R.head(6).itertuples()) + ". Unresolved: " + (", ".join(X["sched"][X["sched"].status == "Unresolved"].machine_id) or "none")))
    if re.search(r"cost|scenario", ql): m_ = mid or (R.machine_id.iloc[0] if len(R) else None); r = R[R.machine_id == m_]
    if re.search(r"cost|scenario", ql) and len(r): T = sim_rows(dict(p=.15, eff=.7, cph=1500.0, hz=X["P"]["hz"], maint_h=float(r.duration_h.iloc[0]), maint_cost=float(r.cost.iloc[0]), fail_h=24.0, repair=5000.0, wait=7.0, exped=400.0, saved=7.0, shift=600.0, defer=14.0, share=1.0)); parts.append(("cost_scenarios", "Hypothetical for %s (ASSUMED p=15%%, $1,500/h downtime, 24 h failure): " % m_ + "; ".join("%s %s" % (a, money(b)) for a, b in zip(T.scenario, T.total)) + ". Adjust assumptions in the Cost & Scenario Lab."))
    if re.search(r"limit|caveat|trust", ql): parts.append(("limitations", "Anomalies are not failures; health is not a probability; the optional model uses %s; RUL is disabled; the schedule is a heuristic (no OR-Tools); there is no live telemetry; every action needs qualified human review." % ("synthetic labels" if G.get("demo") else "your labels")))
    if not parts: parts = [("summary", "%d machines: %s. %d equipment anomalies, %d sensor/data issues, %d proposed actions (%d unresolved)." % (len(mt), ", ".join("%s %d" % (s, int((mt.state == s).sum())) for s in STATES), int((X["an"].cat == "equipment").sum()), int((X["an"].cat == "sensor_fault").sum()), len(R), int((X["sched"].status == "Unresolved").sum())))]
    ctx = "\n".join("[%s] %s" % (n, t) for n, t in parts); out = "### Answer (deterministic, computed by analytical tools)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Every figure comes from the loaded dataset or is labelled hypothetical. Qualified human review is required.*"
    if not use_llm: return out
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k: return out + "\n\n*External LLM disabled: no ANTHROPIC_API_KEY environment variable is set. The answer above is complete.*"
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": k, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json={"model": "claude-sonnet-5-5", "max_tokens": 700, "system": "You are a maintenance analytics assistant. Answer ONLY from the TOOL RESULTS. Never invent machine histories, diagnoses or numbers. Be concise.", "messages": [{"role": "user", "content": "TOOL RESULTS:\n" + ctx + "\n\nQUESTION: " + q}]}, timeout=60); r.raise_for_status()
        return "### LLM-written answer (aggregated tool results sent after you ticked the box)\n" + "".join(b.get("text", "") for b in r.json().get("content", [])) + "\n\n---\n" + out
    except Exception as e: return out + "\n\n*LLM call failed (%s); deterministic answer shown.*" % type(e).__name__

# ------------------------------ exports ------------------------------
def export_all():
    if "X" not in G: return None, note("Load data first.", "info")
    X = G["X"]; out = []
    def w(df, fn):
        df.to_csv(fn, index=False)
        if os.path.exists(fn): out.append(fn)
    mt = X["mt"]; w(mt.drop(columns=["expl"]).assign(explanation=mt.expl), "sentinel_machine_health.csv"); w(X["an"], "sentinel_anomaly_events.csv"); w(mt[["machine_id", "etype", "n_fail", "mtbf", "mttr", "repeat", "n_unpl", "down", "cost"]], "sentinel_failure_patterns.csv"); w(X["recs"], "sentinel_maintenance_recommendations.csv"); w(X["sched"], "sentinel_proposed_schedule.csv")
    R = X["recs"]; pt = X["D"]["parts"]; w(R[R.part_id.notna()].merge(pt, on="part_id", how="left") if len(R) and len(pt) else pd.DataFrame(), "sentinel_spare_part_requirements.csv"); w(X["px"], "sentinel_production_exposure.csv")
    T = sim_rows(dict(p=.15, eff=.7, cph=1500.0, hz=X["P"]["hz"], maint_h=4.0, maint_cost=500.0, fail_h=24.0, repair=5000.0, wait=7.0, exped=400.0, saved=7.0, shift=600.0, defer=14.0, share=1.0)); w(T.assign(assumptions="example assumptions: p=15%, 1500/h, 24 h failure, 70% effectiveness"), "sentinel_scenario_comparison_example.csv")
    top = mt[mt.state != "Normal"].sort_values("priority", ascending=False).head(6); un = X["sched"][X["sched"].status == "Unresolved"]
    md = ["# SENTINEL AI — Executive report", "", "_Synthetic data with planted patterns; no live telemetry._" if G.get("demo") else "", "", "## Highest-priority machine risks (health is not a failure probability)"] + ["- %s (%s, health %.0f): %s. %s" % (r.machine_id, r.state, r.health, r.issue, r.expl) for r in top.itertuples()] + ["", "## Recommended actions (pending qualified human review)"] + ["- %s %s: %s; %s h, skill %s%s, est. cost %s." % (r.rec_id, r.machine_id, r.action, r.duration_h, r.skill, (", part " + r.part_id) if not nul(r.part_id) else "", money(r.cost)) for r in R.itertuples()] + ["", "## Proposed windows"] + ["- %s %s: %s (%s)" % (r.machine_id, dts(r.start), r.status, r.reason) for r in X["sched"].itertuples() if r.status != "Unresolved"] + ["", "## Production impact", "%d non-normal machines have %d dependent orders in the next 30 days (see sentinel_production_exposure.csv)." % (len(X["px"]), int(X["px"].n_orders.sum())), "", "## Unresolved risks needing human decisions"] + (["- %s (%s): %s" % (r.machine_id, r.rec_id, r.reason) for r in un.itertuples()] or ["- none"]) + ["", "## Assumptions and limitations", "Anomalies are not failures. The scheduler is a heuristic without OR-Tools. RUL is not estimated. Scenario costs depend on user assumptions. No PDF export in this build."]
    open("sentinel_executive_report.md", "w", encoding="utf-8").write("\n".join(md)); out.append("sentinel_executive_report.md"); css = "body{font-family:Inter,Arial,sans-serif;background:#16232B;color:#F1F5F4;max-width:980px;margin:30px auto;padding:0 18px}h1{color:#80E0C0}"
    open("sentinel_executive_report.html", "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>SENTINEL AI</title><style>" + css + "</style></head><body><h1>SENTINEL AI — Executive report</h1>" + "".join("<p>%s</p>" % esc(x) for x in md[2:] if x) + "</body></html>"); out.append("sentinel_executive_report.html")
    return [f for f in out if os.path.exists(f)], note("Generated %d files from the active dataset." % len(out), "ok")

# ------------------------------ self-tests ------------------------------
def mini():
    r = np.random.default_rng(1); T = 160; tm = pd.Timestamp("2026-03-01") - pd.to_timedelta(20 * (T - 1 - np.arange(T)), unit="h"); ser = lambda b, s: b * (1 + s * r.normal(size=T)); v1 = ser(2.0, .05); v1[-20:] *= 1.8; v2 = ser(2.0, .05); v2[-14:] = v2[-15]
    sen = pd.concat([pd.DataFrame(dict(timestamp=tm, machine_id=m, vibration=v, temperature=t, runtime_hours=np.arange(T) * 20.0 + 1000, operating_state="Running")) for m, v, t in (("A", v1, np.full(T, 50.0)), ("B", v2, ser(60, .02)), ("C", ser(2.0, .05), ser(55, .02)))], ignore_index=True)
    mach = pd.DataFrame(dict(machine_id=["A", "B", "C"], machine_name=["a", "b", "c"], equipment_type=["Pump"] * 3, production_line=["L1"] * 3, criticality=[3, 3, 3], maintenance_policy=["interval-based"] * 3, pm_interval_hours=[2400.0] * 3))
    wo = pd.DataFrame(dict(work_order_id=["W1", "W2"], machine_id=["C", "C"], maintenance_date=[tm[62], tm[122]], maintenance_type=["Corrective", "Corrective"], downtime_hours=[10.0, 20.0], repair_cost=[1000.0, 2000.0], parts_cost=[100.0, 100.0]))
    fe = pd.DataFrame(dict(event_id=["E1", "E2"], machine_id=["C", "C"], failure_timestamp=[tm[60], tm[120]], failure_type=["Bearing wear"] * 2, downtime_hours=[10.0, 20.0], failure_confirmed=[True, True]))
    pr = pd.DataFrame(dict(production_order_id=["O1", "O2"], machine_id=["A", "B"], planned_start=[tm[-1] + Td(1), tm[-1] + Td(1)], planned_end=[tm[-1] + Td(4), tm[-1] + Td(4)], output_per_hour=[10.0, np.nan], machine_dependency=["True", "True"]))
    res = pd.DataFrame(dict(technician_id=["T1", "T2"], skill_category=["Mechanical", "Specialist"], available_hours=[8.0, 8.0], planned_assignments=[0.0, 6.0])); parts = pd.DataFrame(dict(part_id=["P01"], part_name=["Bearing set"], compatible_machine_types=["Pump"], quantity_available=[0.0], reorder_lead_time_days=[10.0], unit_cost=[100.0]))
    return dict(mach=mach, sen=sen, wo=wo, prod=pr, fe=fe, res=res, parts=parts, dtw=None)
def run_selftests(verbose=True):
    if TESTS: return TESTS
    T = []; saved = dict(G)
    def chk(g, n, fn):
        try: ok, det = fn(); T.append((g, n, bool(ok), det))
        except Exception as e: T.append((g, n, False, type(e).__name__ + ": " + str(e)[:90]))
    MX = lambda: build(load_all(mini()), dict(DEF)); DXd = lambda: DX()
    def t_v1():
        r = mini(); r["mach"] = r["mach"].drop(columns=["criticality"])
        try: load_all(r); return False, "no error"
        except ValueError as e: return "criticality" in str(e), "missing required column reported with guidance"
    def t_v2():
        r = mini(); s = r["sen"]; r["sen"] = pd.concat([s, s.iloc[[0]], s.iloc[[1]].assign(machine_id="ZZ")], ignore_index=True); r["sen"].loc[5, "vibration"] = 500.0; D = load_all(r); w = " | ".join(m for _, m in D["issues"])
        return "duplicate" in w and "unknown machine_id" in w and "outside the plausible range" in w and pd.isna(D["sen"].query("machine_id=='A'").vibration.iloc[5]), "duplicate, unknown machine and out-of-range value (set to missing, not replaced) all reported"
    def t_rb(): m, s = robust(np.full(50, 5.0)); return m == 5.0 and s > 0 and np.isnan(robust([1, 2])[0]), "constant channel gets a positive scale floor; too few points give no baseline"
    def t_an():
        X = MX(); a = X["an"]; ea = a[(a.machine_id == "A") & (a.cat == "equipment")]; fb = a[(a.machine_id == "B") & (a.cat == "sensor_fault")]; eb = a[(a.machine_id == "B") & (a.cat == "equipment") & (a.sensor == "vibration")]; ec = a[(a.machine_id == "C") & (a.cat == "equipment")]
        return len(ea) > 0 and ea.kind.iloc[0] == "Persistent excursion" and len(fb) > 0 and len(eb) == 0 and len(ec) == 0, "A: persistent excursion; B: flatline classed as sensor fault, not equipment; C: no false alert"
    def t_h():
        X = MX(); mt = X["mt"].set_index("machine_id"); return mt.health.between(0, 100).all() and mt.health["A"] < mt.health["C"] and set(mt.state) <= set(STATES), "health within 0-100; shifted machine A (%.0f) below healthy C (%.0f)" % (mt.health["A"], mt.health["C"])
    def t_mb():
        X = MX(); mt = X["mt"].set_index("machine_id"); return mt.n_fail["C"] == 2 and abs(mt.mtbf["C"] - 1590) < 1 and abs(mt.mttr["C"] - 15) < 1e-9 and np.isnan(mt.mtbf["A"]), "MTBF 1590 h and MTTR 15 h for C; A has no failures so MTBF is censored (not zero)"
    def t_px():
        X = MX(); a = prod_for(X, "A", X["asof"] + Td(1), 24); b = prod_for(X, "B", X["asof"] + Td(1), 24); c = prod_for(X, "C", X["asof"] + Td(1), 24); return abs(a["lost"] - 240) < 1e-6 and np.isnan(b["lost"]) and c["n"] == 0 and c["lost"] == 0, "24 h × 10/h = 240 units; missing rate gives 'not estimated'; no orders gives 0"
    def t_sc():
        X = MX(); R = pd.DataFrame([dict(rec_id="R1", machine_id="A", action="x", state="Inspection recommended", urgency_days=3, duration_h=5.0, skill="Mechanical", part_id=None, part_consumed=False, priority=10.0), dict(rec_id="R2", machine_id="A", action="y", state="Inspection recommended", urgency_days=3, duration_h=5.0, skill="Mechanical", part_id=None, part_consumed=False, priority=9.0),
                                    dict(rec_id="R3", machine_id="B", action="z", state="Urgent review", urgency_days=3, duration_h=4.0, skill="Specialist", part_id=None, part_consumed=False, priority=8.0), dict(rec_id="R4", machine_id="B", action="w", state="Inspection recommended", urgency_days=3, duration_h=2.0, skill="Mechanical", part_id="P01", part_consumed=True, priority=7.0)])
        S = schedule(X, R).set_index("rec_id"); return S.day["R1"] != S.day["R2"] and S.status["R3"] == "Unresolved" and "free" in S.reason["R3"] and S.day["R4"] >= 10 and S.status["R4"] == "Scheduled after target deadline", "no technician double-booking (5 h jobs on different days); Specialist 4 h > 2 h free is reported unresolved; missing part delays to day %d" % S.day["R4"]
    def t_sm():
        q = dict(p=.2, eff=.7, cph=0.0, hz=14, maint_h=4.0, maint_cost=0.0, fail_h=24.0, repair=0.0, wait=7.0, exped=0.0, saved=0.0, shift=0.0, defer=14.0, share=1.0); z = sim_rows(q); a = sim_rows(dict(q, p=.1, cph=100.0, repair=1000.0)).total.iloc[0]; b = sim_rows(dict(q, p=.4, cph=100.0, repair=1000.0)).total.iloc[0]
        return (z.total == 0).all() and b > a, "zero costs give zero totals; continue-monitoring cost rises with p (%.0f -> %.0f)" % (a, b)
    def t_md():
        X = MX(); M = X["model"]; return (not M["ok"]) and "needs at least" in M["why"], "mini data has too few failures: model disabled with an explanation"
    def t_fm():
        X = DXd(); M = X["model"]; return M["ok"] and M["ap"] > M["prev"], "demo model PR-AUC %.2f vs prevalence %.3f and baseline %.2f (synthetic labels)" % (M["ap"], M["prev"], M["ap_base"]) if M["ok"] else M.get("why", "")
    def t_dm():
        X = DXd(); mt = X["mt"].set_index("machine_id"); a = X["an"]; ok = mt.health.between(0, 100).all() and len(X["D"]["sen"]) >= 20000 and mt.state["M07"] in ("Urgent review", "Inspection recommended") and (a[(a.machine_id == "M33") & (a.cat == "sensor_fault")].shape[0] > 0) and (a[(a.machine_id == "M45") & (a.cat == "sensor_fault")].shape[0] > 0)
        return ok, "demo: M07 (rising vibration) %s; M33 flatline and M45 dropout classed as sensor faults; %d sensor rows" % (mt.state["M07"], len(X["D"]["sen"]))
    def t_x():
        s2 = dict(G); X = DXd(); G.clear(); G.update(X=X, D=X["D"], demo=True); f, _ = export_all(); n = len(pd.read_csv("sentinel_machine_health.csv")); G.clear(); G.update(s2); return n == len(X["mt"]) and "sentinel_executive_report.md" in f, "%d files; machine health export has %d rows matching the analysis" % (len(f), n)
    def t_l(): s2 = dict(G); G.clear(); G.update(X=MX(), D=None, demo=False); a = ask("Which machines need inspection first?", False); G.clear(); G.update(s2); return "deterministic" in a, "strategist answers from tools without any API key"
    for g, n, f in [("validation", "missing required column", t_v1), ("validation", "duplicates, unknown ids, out-of-range values", t_v2), ("features", "robust baseline edge cases", t_rb), ("anomaly", "persistent excursion vs sensor fault", t_an), ("health", "score bounds and ordering", t_h), ("maintenance", "MTBF censoring and MTTR", t_mb), ("production", "impact and missing rates", t_px),
                    ("scheduler", "technician, skill and part constraints", t_sc), ("simulator", "formulas and monotonicity", t_sm), ("failure", "model disabled when data is insufficient", t_md), ("failure", "demo model beats chance (synthetic)", t_fm), ("health", "planted demo patterns", t_dm), ("reporting", "CSV consistency", t_x), ("strategist", "works without an API key", t_l)]: chk(g, n, f)
    G.clear(); G.update(saved); TESTS[:] = T
    if verbose:
        for g, n, ok, d in T: print(("PASS " if ok else "FAIL ") + g + ": " + n + " :: " + d)
        print("Result: %d of %d self-tests passed." % (sum(1 for t in T if t[2]), len(T)))
    return TESTS
def DX():
    if "X" not in _DEMO: _DEMO["X"] = build(get_demo(), dict(DEF))
    return _DEMO["X"]

# ------------------------------ project export ------------------------------
METH = """### Methodology
- **Baselines:** per machine and channel, median and robust scale (1.4826 × MAD, with a small floor) of the first N running observations. Idle and down observations are excluded from anomaly logic.
- **Anomalies:** persistent excursion (recent window mostly beyond the z threshold), gradual deterioration (|z| ≥ 2 and a rising trend), sudden spike, multivariate Isolation Forest per equipment type, flatline/dropout/stale data classed as sensor faults. An anomaly is not a failure.
- **Health score (0-100):** 100 × (1 − Σ weight × risk ÷ Σ weights) over sensor excursions, trend, failure recurrence, service interval (reduced for policy-deferrable machines) and data quality; missing factors are dropped and named. Not a probability. States: ≥80 Normal, ≥65 Watch, ≥45 Inspection recommended, else Urgent review; criticality ≥ 4 with a severe excursion escalates one level; 3+ failures or sensor faults floor at Watch.
- **Failure model (optional):** target = confirmed failure within the horizon; class-weighted logistic regression; machine-grouped cross-validation; baseline comparison; scores are rankings, not calibrated probabilities. Demo labels are synthetic.
- **RUL:** disabled (needs run-to-failure data).
- **Scheduling:** deterministic priority-ordered heuristic (no OR-Tools) with technician, skill, part and production-window constraints.
- **Simulator:** expected loss = probability (user assumption) × failure cost, with explicit wait-time and effectiveness assumptions; hypothetical.
- **Not included:** OR-Tools, PDF export, live telemetry, multi-day jobs, frequency-domain features.
"""
LICENSE = "MIT License\n\nCopyright (c) 2026 SENTINEL AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# SENTINEL AI — Predictive Maintenance & Equipment Reliability Intelligence
Detect the signal. Prevent the shutdown.

Analyses machine sensor readings, maintenance history and production schedules to find emerging equipment risks, links them to production exposure and proposes maintenance actions with a constraint-aware schedule. Bundled data is **fictional synthetic data**; there is no live telemetry.

## Architecture
@@@mermaid
flowchart LR
  A[CSV or synthetic data] --> B[Validation + unit checks]
  B --> C[Machine baselines]
  C --> D[Anomaly detection + sensor-fault separation]
  D --> E[Health score + states]
  B --> F[Failure and maintenance analytics]
  F --> G[Optional failure model]
  E --> H[Recommendations]
  H --> I[Constraint-aware scheduler]
  I --> J[Production impact + cost simulator]
  J --> K[Strategist, exports, report]
@@@

## Run
@@@
pip install -r requirements.txt
python app.py
@@@
In Google Colab paste `app.py` into one cell. The `src/` modules in the brief are consolidated into `app.py`.

## Tests, docs, safety
`pip install pytest && pytest tests` runs the built-in self-tests. See `docs/`. Optional LLM: set `ANTHROPIC_API_KEY` (see `.env.example`); off by default. Not included: OR-Tools, RUL, PDF. Decision support only: no maintenance is executed and safety-critical decisions need qualified engineers. Add screenshots to `screenshots/`.
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] in %s]\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "SENTINEL AI — Predictive Maintenance" in s and "def main" in s: return s
    except Exception: pass
    return None
def write_samples():
    raw = get_demo_raw(); os.makedirs("sample_data", exist_ok=True); out = []; fn = dict(mach="machines", sen="sensor_readings", wo="maintenance_history", fe="failure_events", prod="production_schedule", res="maintenance_resources", parts="spare_parts", dtw="planned_downtime")
    for k, n in fn.items(): p = "sample_data/%s.csv" % n; raw[k].to_csv(p, index=False); out.append(p)
    return out
def build_project(files):
    root = "sentinel-ai-predictive-maintenance"
    for d in ["sample_data", "tests", "docs", "screenshots"]: os.makedirs(os.path.join(root, d), exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    for f in files: open(os.path.join(root, f), "wb").write(open(f, "rb").read())
    W = {"requirements.txt": "gradio>=4.44,<6\nplotly\nscipy\nscikit-learn\nrequests\npandas\nnumpy\n", "README.md": README, "LICENSE": LICENSE, ".env.example": "# Optional. Leave empty to keep the external LLM disabled.\nANTHROPIC_API_KEY=\n", ".gitignore": ".env\n*.key\nsecrets*\n__pycache__/\n.ipynb_checkpoints/\n*.zip\nsentinel_*.csv\nsentinel_*.md\nsentinel_*.html\n",
         "docs/methodology.md": METH, "docs/data_dictionary.md": "# Data dictionary\nmachines, sensor_readings (units: degC, mm/s, bar, rpm, A, %), maintenance_history, failure_events, production_schedule (output_per_hour, machine_dependency, alternate_machine_id), maintenance_resources, spare_parts, planned_downtime. All samples are synthetic.\n",
         "docs/model_evaluation.md": "# Model evaluation\nTarget: confirmed failure within the horizon. Machine-grouped cross-validation, PR-AUC vs a simple deviation baseline, precision/recall/F1, confusion matrix, Brier. Scores are not calibrated probabilities. Demo labels are synthetic.\n",
         "docs/maintenance_scheduling.md": "# Scheduling\nHeuristic: urgency then risk; technician free hours, skills, part stock and production days are constraints; unplaceable tasks are reported. Not OR-Tools.\n", "docs/responsible_ai.md": "# Responsible use\nNo anomaly is a confirmed failure. No unvalidated probability is a guarantee. RUL is never invented. Nothing is executed automatically; safety-critical actions need qualified human review.\n",
         "docs/limitations.md": "# Limitations\nSynthetic data; no live telemetry; heuristic scheduler; one-day tasks; RUL disabled; baseline assumes the first observations are healthy.\n", "screenshots/README.md": "# Screenshots\nAdd your own screenshots here. None are included.\n"}
    for nm, g in [("data_validation", "validation"), ("sensor_features", "features"), ("machine_health", "health"), ("anomaly_detection", "anomaly"), ("failure_prediction", "failure"), ("maintenance_recommender", "maintenance"), ("maintenance_scheduler", "scheduler"), ("scenario_simulator", "simulator")]: W["tests/test_%s.py" % nm] = TFILE % (nm, "('%s',)" % g)
    for fn, tx in W.items(): open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "sentinel_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#16232B!important;color:#F1F5F4!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1380px!important}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:14px;padding:26px 30px;border-radius:14px;background:#253640;border:1px solid #ffffff1a;border-left:4px solid #80E0C0;margin-bottom:12px}
.brand{display:flex;gap:16px;align-items:center}.logo{width:50px;height:50px;border-radius:10px;border:2px solid #80E0C0;color:#80E0C0;font-weight:800;font-size:20px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:30px;letter-spacing:3px;color:#F1F5F4}.hdr h1 span{color:#80E0C0}.hdr .sub{color:#D2B8FF;font-size:13px;letter-spacing:2px;text-transform:uppercase}.hdr p{margin:6px 0 0;color:#A5B6C0;font-size:15px;font-style:italic}
.badges span{display:inline-block;margin:3px;padding:4px 11px;border-radius:6px;background:#ffffff0d;border:1px solid #ffffff1a;color:#A5B6C0;font-size:12px}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:10px 0}.kc{background:#253640;border:1px solid #ffffff14;border-radius:10px;padding:14px 16px}
.kl{color:#A5B6C0;font-size:11px;text-transform:uppercase;letter-spacing:1.2px}.kv{font-size:24px;font-weight:700;margin:6px 0 4px}.kd{color:#A5B6C0;font-size:12px}
.note{background:#253640;padding:10px 14px;border-radius:8px;margin:8px 0;color:#F1F5F4;font-size:14px;line-height:1.5}.sec{font-size:15px;font-weight:700;color:#D2B8FF;margin:18px 0 6px;text-transform:uppercase;letter-spacing:1px}
.pill{display:inline-block;border:1px solid;border-radius:99px;padding:1px 9px;font-size:11px;font-weight:600;margin:1px}.tw{overflow-x:auto;border-radius:8px;border:1px solid #ffffff14;margin:8px 0}table{border-collapse:collapse;width:100%;background:#253640;font-size:13px}
th{background:#30464f;color:#80E0C0;text-align:left;padding:9px 10px;font-weight:600}td{padding:8px 10px;border-top:1px solid #ffffff10;color:#F1F5F4;vertical-align:top}tr:hover td{background:#80E0C00d}
button.primary{background:#80E0C0!important;color:#16232B!important;border:none!important;font-weight:700!important}@media(max-width:640px){.hdr h1{font-size:22px}.kv{font-size:20px}.hdr{padding:18px}}
"""
HEADER = ("<div class='hdr'><div class='brand'><div class='logo'>SN</div><div><h1>SENTINEL <span>AI</span></h1><div class='sub'>Predictive Maintenance &amp; Equipment Reliability Intelligence</div><p>Detect the signal. Prevent the shutdown.</p></div></div>"
          "<div class='badges'><span>Anomaly ≠ failure</span><span>Health ≠ probability</span><span>Scenarios are hypothetical</span><span>Qualified human review required</span><span>No live telemetry</span></div></div>")
SK = ["base_n", "win", "zthr", "persist", "hz", "labour", "w_sensor", "w_trend", "w_fail", "w_service", "w_dq"]
def mkP(*v):
    P = dict(DEF); d = dict(zip(SK, v)); P["w"] = {k[2:]: d.pop(k) for k in [x for x in SK if x.startswith("w_")]}; P.update(d); P["base_n"] = int(P["base_n"]); P["win"] = int(P["win"]); P["hz"] = int(P["hz"]); return P
def run_all(*a):
    n = len(SK)
    try:
        P = mkP(*a[:n]); files = a[n:n + 8]; mapping = a[n + 8]; demo = files[0] is None
        D = get_demo() if demo else load_all(dict(zip(TN.keys(), [read_csv_any(f) if f is not None else None for f in files])), mapping); X = DX() if (demo and P == DEF) else build(D, P); G.clear(); G.update(X=X, D=D, P=P, demo=demo)
        return note(("Fictional demo data analysed. " if demo else "Your data analysed. ") + "%d machines, %s sensor rows; %d equipment anomalies; %d proposed actions." % (len(D["mach"]), nf(len(D["sen"])), int((X["an"].cat == "equipment").sum()), len(X["recs"])), "ok")
    except Exception as e: return note("Could not complete the analysis: " + esc(e), "err")
def refresh():
    if "X" not in G: return tuple(gr.update() for _ in range(7))
    X = G["X"]; mt = X["mt"]; ids = list(mt.sort_values("priority", ascending=False).machine_id); top = ids[0] if ids else None
    return (gr.update(choices=sorted(mt.etype.unique()), value=[]), gr.update(choices=sorted(mt.line.unique()), value=[]), gr.update(choices=ids, value=top), gr.update(choices=ids, value=top), gr.update(choices=ids, value=top), gr.update(choices=ids, value=top), gr.update(choices=CH, value="vibration"))
def build_ui(zip_path, sfiles):
    with gr.Blocks(css=CSS, title="SENTINEL AI", theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("Settings (press Recalculate after changing)", open=False):
            with gr.Row():
                s1 = gr.Slider(40, 200, 100, step=10, label="Baseline: first N running observations"); s2 = gr.Slider(4, 20, 8, step=1, label="Recent window (observations)"); s3 = gr.Slider(2, 6, 3.5, step=.5, label="Anomaly threshold (robust z)"); s4 = gr.Slider(.3, 1, .6, step=.1, label="Persistence required"); s5 = gr.Slider(7, 30, 14, step=1, label="Failure horizon (days)"); s6 = gr.Number(60, label="Labour cost per hour")
            gr.Markdown("**Health-score weights** (normalised automatically)")
            with gr.Row():
                w1 = gr.Slider(0, 1, .4, step=.05, label="Sensor excursions"); w2 = gr.Slider(0, 1, .15, step=.05, label="Trend"); w3 = gr.Slider(0, 1, .15, step=.05, label="Failure recurrence"); w4 = gr.Slider(0, 1, .15, step=.05, label="Service interval"); w5 = gr.Slider(0, 1, .15, step=.05, label="Data quality")
            run_btn = gr.Button("Recalculate", variant="primary")
        with gr.Accordion("Filters (equipment type and production line)", open=True):
            with gr.Row():
                f_t = gr.Dropdown([], multiselect=True, label="Equipment type"); f_l = gr.Dropdown([], multiselect=True, label="Production line"); ap_btn = gr.Button("Apply filters", variant="primary")
        status = gr.HTML(note("Loading the fictional demo plant and running self-tests (about 1-2 minutes)...", "info")); FIN = [f_t, f_l]
        with gr.Tabs():
            with gr.Tab("01 Fleet Command Centre"):
                fl_h = gr.HTML()
                with gr.Row():
                    fl1 = gr.Plot(); fl2 = gr.Plot()
                fl3 = gr.Plot()
            with gr.Tab("02 Equipment Health"):
                with gr.Row():
                    h_q = gr.Textbox(label="Search machine"); h_mid = gr.Dropdown([], label="Machine drill-down"); h_btn = gr.Button("Update", variant="primary")
                hl_h = gr.HTML()
                with gr.Row():
                    hl1 = gr.Plot(); hl2 = gr.Plot()
            with gr.Tab("03 Sensor Intelligence"):
                with gr.Row():
                    s_mid = gr.Dropdown([], label="Machine"); s_ch = gr.Dropdown(CH, value="vibration", label="Sensor"); s_btn = gr.Button("Show", variant="primary")
                se_h = gr.HTML(); se1 = gr.Plot()
                with gr.Row():
                    se2 = gr.Plot(); se3 = gr.Plot()
            with gr.Tab("04 Failure Intelligence"):
                with gr.Row():
                    fa_mid = gr.Dropdown([], label="Machine timeline"); fa_btn = gr.Button("Update", variant="primary")
                fa_h = gr.HTML()
                with gr.Row():
                    fa1 = gr.Plot(); fa2 = gr.Plot()
                fa3 = gr.Plot()
            with gr.Tab("05 Maintenance Planner"):
                mp_h = gr.HTML(); mp1 = gr.Plot()
                with gr.Row():
                    mp2 = gr.Plot(); mp3 = gr.Plot()
            with gr.Tab("06 Production Impact"):
                pi_h = gr.HTML(); pi1 = gr.Plot()
            with gr.Tab("07 Cost & Scenario Lab"):
                gr.Markdown("All outputs are **hypothetical**. The failure probability is **your assumption**, not a validated value. Leave a field at 0 to use the data-derived default.")
                with gr.Row():
                    sm_mid = gr.Dropdown([], label="Machine"); sm_p = gr.Slider(0, 1, .15, step=.01, label="Assumed failure probability within horizon"); sm_eff = gr.Slider(0, 1, .7, step=.05, label="Maintenance effectiveness (risk removed)"); sm_cph = gr.Number(1500, label="Cost per downtime hour")
                with gr.Row():
                    sm_fh = gr.Number(0, label="Failure downtime h (0 = auto)"); sm_rep = gr.Number(0, label="Unplanned repair cost (0 = auto)"); sm_mh = gr.Number(0, label="Maintenance hours (0 = auto)"); sm_mc = gr.Number(0, label="Maintenance cost (0 = auto)"); sm_wait = gr.Number(0, label="Days to next planned downtime (0 = auto)")
                with gr.Row():
                    sm_ex = gr.Number(400, label="Expedite cost"); sm_sv = gr.Number(7, label="Days saved by expediting"); sm_sh = gr.Number(600, label="Extra shift cost"); sm_df = gr.Number(14, label="Permitted deferral (days)"); sm_share = gr.Slider(0, 1, 1, step=.1, label="Share of maintenance hours that stop production")
                sm_btn = gr.Button("Run scenarios", variant="primary"); sm_h = gr.HTML()
                with gr.Row():
                    sm1 = gr.Plot(); sm2 = gr.Plot()
            with gr.Tab("08 Reports & Data"):
                gr.Markdown("Upload CSVs (machines, sensor readings, maintenance history and production schedule required; failure events, resources, spare parts and planned downtime optional). Leave empty for the fictional demo. Raw uploads are never modified.")
                up = []
                for i in range(0, 8, 4):
                    with gr.Row():
                        for k in list(TN)[i:i + 4]: up.append(gr.File(label=TN[k] + ".csv" + ("" if k in REQ else " (optional)"), file_types=[".csv"]))
                mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="machine_id=asset, vibration=vib_mm_s", lines=2); ld_btn = gr.Button("Validate and load uploaded data", variant="primary")
                gr.File(value=sfiles, label="Sample CSV templates (synthetic)", file_count="multiple", interactive=False); dq_h = gr.HTML(); t_h = gr.HTML()
                gr.Markdown("### Exports"); xp_btn = gr.Button("Generate exports and executive report", variant="primary"); xp_msg = gr.HTML(); xp_files = gr.File(label="Exports", file_count="multiple", interactive=False); gr.File(value=zip_path, label="Download GitHub-ready project (ZIP)", interactive=False)
            with gr.Tab("09 Methodology & Trust"):
                gr.Markdown(METH)
            with gr.Tab("SENTINEL Strategist"):
                gr.Markdown("Answers come from analytical tools over the current data. Try: *Which machines need inspection first?* · *Why is M07 classified as high risk?* · *Which sensor readings have changed most?* · *What failures recur most?* · *Which machines could disrupt next week's production?* · *Should M15 be maintained now or during planned downtime?* · *Which parts and technicians are needed?* · *How do the cost scenarios compare for M07?*")
                ag_q = gr.Textbox(label="Ask a question"); ag_llm = gr.Checkbox(False, label="Use external LLM (off by default; sends only aggregated results; needs ANTHROPIC_API_KEY in the environment; check your data policy first)"); ag_btn = gr.Button("Ask", variant="primary"); ag_out = gr.Markdown()
        RUN = [s1, s2, s3, s4, s5, s6, w1, w2, w3, w4, w5] + up + [mp]; SMIN = [sm_mid, sm_p, sm_eff, sm_cph, sm_fh, sm_rep, sm_mh, sm_mc, sm_wait, sm_ex, sm_sv, sm_sh, sm_df, sm_share]; REF = [f_t, f_l, h_mid, s_mid, fa_mid, sm_mid, s_ch]
        def pages(e):
            e = e.then(render_fleet, FIN, [fl_h, fl1, fl2, fl3]).then(render_health, FIN + [h_q, h_mid], [hl_h, hl1, hl2]).then(render_sensor, [s_mid, s_ch], [se_h, se1, se2, se3]).then(render_fail, fa_mid, [fa_h, fa1, fa2, fa3])
            return e.then(render_plan, FIN, [mp_h, mp1, mp2, mp3]).then(render_prodimp, FIN, [pi_h, pi1]).then(render_sim, SMIN, [sm_h, sm1, sm2]).then(render_dq, None, [dq_h, t_h])
        def full(ev): return pages(ev.then(refresh, None, REF))
        full(run_btn.click(run_all, RUN, [status], show_progress="full")); full(ld_btn.click(run_all, RUN, [status], show_progress="full")); full(demo.load(run_all, RUN, [status]))
        ap_btn.click(render_fleet, FIN, [fl_h, fl1, fl2, fl3]).then(render_health, FIN + [h_q, h_mid], [hl_h, hl1, hl2]).then(render_plan, FIN, [mp_h, mp1, mp2, mp3]).then(render_prodimp, FIN, [pi_h, pi1])
        h_btn.click(render_health, FIN + [h_q, h_mid], [hl_h, hl1, hl2]); s_btn.click(render_sensor, [s_mid, s_ch], [se_h, se1, se2, se3]); fa_btn.click(render_fail, fa_mid, [fa_h, fa1, fa2, fa3]); sm_btn.click(render_sim, SMIN, [sm_h, sm1, sm2], show_progress="full")
        xp_btn.click(export_all, None, [xp_files, xp_msg], show_progress="full"); ag_btn.click(ask, [ag_q, ag_llm], ag_out, show_progress="full"); ag_q.submit(ask, [ag_q, ag_llm], ag_out)
    return demo
def main():
    get_demo(); sf = write_samples(); zp, saved = build_project(sf)
    print("=" * 78); print("SENTINEL AI starting. Demo data is FICTIONAL with planted patterns. Core analytics need no API key."); print("Building the demo analysis and running self-tests (about 1-2 minutes):")
    run_selftests(verbose=True); demo = build_ui(zp, sf)
    print("Project ZIP:", zp, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("Not included / not tested: OR-Tools, RUL estimation, PDF export, uploads of your own CSVs, the optional LLM, live telemetry. The public Gradio link is open to anyone: do not load confidential data unless access is restricted.")
    print("=" * 78); demo.queue().launch(debug=False, show_error=True)
if __name__ == "__main__" or "google.colab" in sys.modules:
    main()