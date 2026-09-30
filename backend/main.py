import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pipeline import build
app = FastAPI(title="SAFEYRA API - DEMO MODE (synthetic data)")
ENG, PROJ = build(); IDX = {p["id"]: p for p in PROJ}

def view(p, full=False):
    r = ENG.risk(p["features"]); tr = p["trajectory"]; d = tr[-1]["risk"] - tr[-2]["risk"]
    v = {**{k: p[k] for k in ("id", "state", "district", "stage_name", "lat", "lon")}, "risk": r["probability"], "category": r["category"],
         "trend": "Rising" if d > .03 else "Falling" if d < -.03 else "Stable", "bottleneck": r["likely_bottleneck"]}
    return {**p, **v, "detail": r} if full else v

def get(pid):
    if pid not in IDX: raise HTTPException(404, "Project not found")
    return IDX[pid]

@app.get("/api/health")
def health(): return {"status": "ok", "mode": "DEMO - synthetic data", "metrics": ENG.metrics}
@app.get("/api/projects")
def projects(): return [view(p) for p in PROJ]
@app.get("/api/projects/{pid}")
def project(pid: str): return view(get(pid), True)
class Sim(BaseModel):
    comp_progress: float = Field(ge=0, le=100); stage_days: float = Field(ge=0, le=1000)
    pending_approvals: int = Field(ge=0, le=50); legal_disputes: int = Field(ge=0, le=50); rr_progress: float = Field(ge=0, le=100)
@app.post("/api/projects/{pid}/simulate")
def simulate(pid: str, s: Sim):
    p = get(pid); cur = ENG.prob(p["features"]); new = ENG.prob({**p["features"], **s.model_dump()})
    return {"current": round(cur, 3), "scenario": round(new, 3), "change_pp": round((new - cur) * 100, 1),
            "note": "Model-predicted change under a hypothetical scenario. Not a causal or guaranteed effect."}
@app.get("/api/alerts")
def alerts():
    out = []
    for p in PROJ:
        t = p["trajectory"]; d = t[-1]["risk"] - t[-2]["risk"]
        if t[-1]["risk"] >= .7: out.append({"type": "HIGH RISK", "id": p["id"], "msg": f"{p['id']} predicted risk {t[-1]['risk']:.0%}"})
        if d >= .1: out.append({"type": "RAPID CHANGE", "id": p["id"], "msg": f"{p['id']} risk {t[-2]['risk']:.0%} -> {t[-1]['risk']:.0%}"})
    return out[:30]
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="ui")
