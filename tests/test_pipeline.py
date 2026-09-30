import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ml"))
from pipeline import build
def test_predictions_in_range():
    eng, proj = build(10)
    for p in proj: assert 0 <= eng.risk(p["features"])["probability"] <= 1
def test_scenario_direction():
    eng, proj = build(10); f = proj[0]["features"]
    assert eng.prob({**f, "comp_progress": 95, "pending_approvals": 0, "legal_disputes": 0}) <= eng.prob({**f, "comp_progress": 5, "pending_approvals": 5, "legal_disputes": 4})
