import xgboost as xgb
import numpy as np
import json
import os

MODEL_PATH = "models/xgb_model.json"
print("MODEL_PATH:", MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model file not found")

m = xgb.Booster()
m.load_model(MODEL_PATH)
FEATURE_NAMES = [
    'points_diff','rank_diff','tournaments_diff',
    'hist_avg_points_diff','hist_std_points_diff','hist_slope_diff','hist_count_diff'
]
def pred_from_inputs(pa, pb, pa_rank, pb_rank, pa_tourn, pb_tourn, pa_hist_avg, pb_hist_avg, pa_hist_std, pb_hist_std, pa_hist_slope, pb_hist_slope, pa_hist_count, pb_hist_count):
    points_diff = pa - pb
    rank_diff = pb_rank - pa_rank
    tournaments_diff = pa_tourn - pb_tourn
    hist_avg_diff = pa_hist_avg - pb_hist_avg
    hist_std_diff = pa_hist_std - pb_hist_std
    hist_slope_diff = pa_hist_slope - pb_hist_slope
    hist_count_diff = pa_hist_count - pb_hist_count
    X = np.array([[points_diff, rank_diff, tournaments_diff, hist_avg_diff, hist_std_diff, hist_slope_diff, hist_count_diff]], dtype=float)
    d = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    p = float(m.predict(d)[0])
    return p

# extreme A> B
p1 = pred_from_inputs(20000,1000,1,100,20,2,18000,900,100,50,100,-50,6,6)
print("p1:", p1)

# extreme B > A (role-flipped)
p2 = pred_from_inputs(1000,20000,100,1,2,20,900,18000,50,100,-50,100,6,6)
print("p2:", p2)
