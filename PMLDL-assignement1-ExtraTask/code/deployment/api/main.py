from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import os
import joblib
from sklearn.calibration import CalibratedClassifierCV

app = FastAPI()
MODEL_DIR = os.environ.get('MODEL_DIR', '/app/models')
CALIBRATED_MODEL_PATH = os.path.join(MODEL_DIR, 'calibrated_xgb_model.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model.json')
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, 'feature_info.pkl')

@app.on_event("startup")
def load_model():
    global model, feature_names, exclude_leaky_features, is_calibrated
    
    try:
        # Load feature info
        if os.path.exists(FEATURE_INFO_PATH):
            feature_info = joblib.load(FEATURE_INFO_PATH)
            feature_names = feature_info['feature_names']
            exclude_leaky_features = feature_info.get('exclude_leaky_features', True)
            is_calibrated = feature_info.get('is_calibrated', False)
        else:
            # Fallback to default features
            feature_names = [
                'rank_diff', 'tournaments_diff',
                'hist_avg_points_diff', 'hist_std_points_diff', 
                'hist_slope_diff', 'hist_count_diff'
            ]
            exclude_leaky_features = True
            is_calibrated = False
        
        # Load the appropriate model
        if is_calibrated and os.path.exists(CALIBRATED_MODEL_PATH):
            model = joblib.load(CALIBRATED_MODEL_PATH)
            print("Loaded calibrated model")
        elif os.path.exists(XGB_MODEL_PATH):
            model = xgb.Booster()
            model.load_model(XGB_MODEL_PATH)
            is_calibrated = False
            print("Loaded XGBoost model")
        else:
            model = None
            print("No model file found")
            
        print(f"Model type: {'calibrated' if is_calibrated else 'XGBoost'}")
        print(f"Features: {feature_names}")
        print(f"Exclude leaky features: {exclude_leaky_features}")
        
    except Exception as e:
        model = None
        print(f"Failed to load model: {e}")

class PredictRequest(BaseModel):
    match_type: str
    player_a_points: float
    player_b_points: float
    player_a_rank: Optional[float] = None
    player_b_rank: Optional[float] = None
    player_a_num_tournaments: int = 0
    player_b_num_tournaments: int = 0

    player_a_hist_avg_points: float = 0.0
    player_b_hist_avg_points: float = 0.0
    player_a_hist_std_points: float = 0.0
    player_b_hist_std_points: float = 0.0
    player_a_hist_slope_points: float = 0.0
    player_b_hist_slope_points: float = 0.0
    player_a_hist_count: int = 0
    player_b_hist_count: int = 0

class PredictResponse(BaseModel):
    winner: str
    probability: float
    confidence: str

def prepare_features(req: PredictRequest, feature_names: list, exclude_leaky_features: bool) -> np.ndarray:
    # Extract inputs
    pa_points = float(req.player_a_points)
    pb_points = float(req.player_b_points)
    pa_rank = float(req.player_a_rank) if req.player_a_rank is not None else 1000.0
    pb_rank = float(req.player_b_rank) if req.player_b_rank is not None else 1000.0
    pa_tourn = int(req.player_a_num_tournaments)
    pb_tourn = int(req.player_b_num_tournaments)

    pa_hist_avg = float(req.player_a_hist_avg_points)
    pb_hist_avg = float(req.player_b_hist_avg_points)
    pa_hist_std = float(req.player_a_hist_std_points)
    pb_hist_std = float(req.player_b_hist_std_points)
    pa_hist_slope = float(req.player_a_hist_slope_points)
    pb_hist_slope = float(req.player_b_hist_slope_points)
    pa_hist_count = int(req.player_a_hist_count)
    pb_hist_count = int(req.player_b_hist_count)

    # Feature mapping
    feature_map = {
        'points_diff': pa_points - pb_points,
        'rank_diff': pb_rank - pa_rank,
        'tournaments_diff': pa_tourn - pb_tourn,
        'hist_avg_points_diff': pa_hist_avg - pb_hist_avg,
        'hist_std_points_diff': pa_hist_std - pb_hist_std,
        'hist_slope_diff': pa_hist_slope - pb_hist_slope,
        'hist_count_diff': pa_hist_count - pb_hist_count,
        'points_ratio': pa_points / (pb_points + 1e-8),
        'rank_ratio': (pb_rank + 1) / (pa_rank + 1e-8)
    }

    # Build feature array in the correct order
    feature_values = []
    for feature_name in feature_names:
        if feature_name in feature_map:
            feature_values.append(feature_map[feature_name])
        else:
            feature_values.append(0.0)  # Default for unknown features

    return np.array([feature_values], dtype=float)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare features
        X = prepare_features(req, feature_names, exclude_leaky_features)
        
        # Make prediction based on model type
        if is_calibrated:
            # Calibrated model uses predict_proba
            prob = model.predict_proba(X)[0, 1]
        else:
            # Regular XGBoost model
            dmat = xgb.DMatrix(X, feature_names=feature_names)
            raw_prob = float(model.predict(dmat)[0])
            prob = raw_prob

        # Apply smoothing to prevent extreme probabilities
        epsilon = 0.01
        prob = max(epsilon, min(1 - epsilon, prob))
        
        # Determine confidence level
        if prob > 0.7 or prob < 0.3:
            confidence = "high"
        elif prob > 0.6 or prob < 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        winner = "player_a" if prob > 0.5 else "player_b"
        return PredictResponse(winner=winner, probability=prob, confidence=confidence)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")