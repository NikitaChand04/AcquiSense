# SAFEYRA
### Predictive Analytics System for Early Detection of Land Acquisition Delays — SIH 2026, SIH26017
**Predict → Explain → Detect Bottleneck → Simulate → Monitor**

> **DEMO MODE.** The prototype uses fully synthetic data (generated in `ml/pipeline.py`). The official SIH26017 dataset was not available, so its schema was not inspected. Demonstration metrics must not be interpreted as real-world performance.

## Run (Python 3.10+; no Node, Docker or PostgreSQL needed)
1. Extract `SAFEYRA-SIH26017.zip`, then `cd SAFEYRA-SIH26017`
2. `python -m venv .venv` → activate (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere)
3. `pip install -r requirements.txt`
4. `uvicorn backend.main:app --reload` (run from the project root)
5. Open http://localhost:8000 (UI) or http://localhost:8000/docs (API docs). Internet is needed for map tiles and chart/map libraries (CDN).
6. Tests: `pytest tests`

## Method
Features: stage, stage duration, compensation progress, pending approvals, legal disputes, R&R progress, documentation gap. Target: delay outcome (synthetic). Baseline: Logistic Regression. Primary: XGBoost, explained with TreeSHAP (`pred_contribs`). If xgboost is not installed, the code falls back to scikit-learn HistGradientBoosting with an *approximate occlusion* explainer, and the UI states this. Validation: chronological 70/15/15 split by observation date; features are snapshot values only (no post-prediction fields). On the synthetic data, logistic regression may match or beat the boosted model, since the synthetic rule is near-logistic.
"Likely bottleneck" = the model's largest risk-raising driver, mapped to a process area (an indicator, not a separate trained model, not a legal determination). What-if = model-predicted change, not causal.

## API
`GET /api/health` · `GET /api/projects` · `GET /api/projects/{id}` (risk, drivers, trajectory) · `POST /api/projects/{id}/simulate` · `GET /api/alerts`

## Limitations
Synthetic data only; single-process in-memory demo; role selection is UI-only (no auth); no PostgreSQL/PostGIS, Docker, or React; no real system integration.
## Sustainability model
Institutional licensing, integration, support, custom analytics and training — never selling land or citizen data.
## Team
SAFEYRA (Team ID 149455)
