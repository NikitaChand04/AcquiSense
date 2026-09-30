"""SAFEYRA demo pipeline. DATA IS SYNTHETIC - metrics are NOT real-world performance."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None
F = ["stage", "stage_days", "comp_progress", "pending_approvals", "legal_disputes", "rr_progress", "doc_gap"]
STAGES = ["Notification", "Land/Process", "Award", "Compensation", "Possession", "R&R/Closure"]
AREA = {"stage_days": "Stage duration", "comp_progress": "Compensation", "pending_approvals": "Approvals",
        "legal_disputes": "Legal disputes", "rr_progress": "R&R", "doc_gap": "Documentation", "stage": "Lifecycle stage"}
sig = lambda z: 1 / (1 + np.exp(-z))

def synth(n=400, seed=7):
    """Synthetic projects, 4 snapshots each. Label = delay outcome drawn from a hidden synthetic rule."""
    r = np.random.default_rng(seed); rows = []
    for i in range(n):
        end = dict(stage=int(r.integers(1, 5)), stage_days=r.uniform(30, 300), comp_progress=r.uniform(10, 95),
                   pending_approvals=int(r.integers(0, 6)), legal_disputes=int(r.integers(0, 5)),
                   rr_progress=r.uniform(5, 90), doc_gap=r.uniform(0, .6))
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(r.integers(0, 500)))
        for t in range(4):
            k = .55 + .15 * t; s = dict(end)
            s.update(comp_progress=end["comp_progress"] * k, rr_progress=end["rr_progress"] * k,
                     stage_days=end["stage_days"] * k, doc_gap=min(1, end["doc_gap"] * (1.6 - .2 * t)))
            z = (-5.6 + .012 * s["stage_days"] + .03 * (60 - s["comp_progress"]) + .45 * s["pending_approvals"]
                 + .55 * s["legal_disputes"] + .015 * (50 - s["rr_progress"]) + 1.5 * s["doc_gap"] + r.normal(0, .6))
            rows.append(dict(project_idx=i, date=base + pd.Timedelta(days=30 * t), **s, delayed=int(r.random() < sig(z))))
    return pd.DataFrame(rows)

class Engine:
    def __init__(self, df):
        df = df.sort_values("date"); a, b = int(len(df) * .7), int(len(df) * .85)   # chronological split
        tr, va, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
        self.med = tr[F].median()
        self.base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(tr[F], tr.delayed)
        if xgb:
            self.model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=.08, eval_metric="logloss")
            self.model.fit(tr[F], tr.delayed, eval_set=[(va[F], va.delayed)], verbose=False)
            self.name, self.explainer = "XGBoost", "TreeSHAP (xgboost pred_contribs)"
        else:
            self.model = HistGradientBoostingClassifier(max_depth=3, learning_rate=.08).fit(tr[F], tr.delayed)
            self.name, self.explainer = "HistGradientBoosting (fallback: xgboost not installed)", "Occlusion vs training median (approximate, not SHAP)"
        self.metrics = {"data": "SYNTHETIC - not real-world performance", "split": "chronological 70/15/15",
                        "test_auc_logreg": round(roc_auc_score(te.delayed, self.base.predict_proba(te[F])[:, 1]), 3),
                        "test_auc_model": round(roc_auc_score(te.delayed, self.model.predict_proba(te[F])[:, 1]), 3),
                        "model": self.name, "explainer": self.explainer}
    def prob(self, feats):
        return float(self.model.predict_proba(pd.DataFrame([feats])[F])[0, 1])
    def explain(self, feats):
        X = pd.DataFrame([feats])[F]
        if xgb:
            c = self.model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
        else:
            lg = lambda p: np.log(p / (1 - p)); p0 = lg(np.clip(self.prob(feats), 1e-4, 1 - 1e-4)); c = []
            for f in F:
                Y = X.copy(); Y[f] = self.med[f]
                c.append(p0 - lg(np.clip(self.model.predict_proba(Y)[0, 1], 1e-4, 1 - 1e-4)))
        return sorted(({"feature": f, "contribution": round(float(v), 3)} for f, v in zip(F, c)), key=lambda d: -abs(d["contribution"]))
    def risk(self, feats):
        p = self.prob(feats); ex = self.explain(feats)
        pos = [e for e in ex if e["contribution"] > 0]
        return {"probability": round(p, 3), "category": "High" if p >= .65 else "Medium" if p >= .4 else "Low",
                "drivers": ex, "likely_bottleneck": AREA[pos[0]["feature"]] if pos else "None indicated",
                "bottleneck_note": "Driver-derived indicator from the model's top risk contributor; not a legal/administrative determination."}

def build(n=120):
    eng = Engine(synth(400)); df = synth(n, seed=99)  # demo projects are unseen by the model
    r = np.random.default_rng(1); out = []
    for i in range(n):
        g = df[df.project_idx == i].sort_values("date"); cur = g.iloc[-1]
        feats = {f: float(cur[f]) for f in F}
        traj = [{"date": str(d.date()), "risk": round(eng.prob({f: float(row[f]) for f in F}), 3)} for d, (_, row) in zip(g.date, g.iterrows())]
        out.append({"id": f"DEMO-{i+1:03d}", "state": "Uttarakhand", "district": f"Demo District {chr(65 + i % 6)}",
                    "project_type": ["Highway", "Rail", "Irrigation", "Power"][i % 4], "land_area_ha": int(r.integers(20, 400)),
                    "families": int(r.integers(20, 500)), "lat": float(29.6 + r.uniform(0, 1.6)), "lon": float(78.0 + r.uniform(0, 2)),
                    "features": feats, "stage_name": STAGES[int(feats["stage"])], "trajectory": traj})
    return eng, out
