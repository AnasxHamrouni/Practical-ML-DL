#!/usr/bin/env python3
"""
Training script with MLflow integration and params.yaml support.

- Reads processed data from --processed-dir (default: data/processed)
- Reads params from params.yaml
- CLI args override params.yaml when the corresponding flag is passed on the command line
- Logs params and metrics to MLflow
- Saves model files and models/metrics.json for DVC to pick up
"""
import os
import sys
import json
import argparse
from pathlib import Path

import joblib
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
import mlflow.xgboost


def load_params_file(root: Path):
    params_path = root / "params.yaml"
    if params_path.exists():
        try:
            with open(params_path, "r") as f:
                content = yaml.safe_load(f) or {}
            return content.get("train", {})
        except Exception as e:
            print("Failed to load params.yaml:", e)
            return {}
    return {}


def param_value(cli_flag: str, param_name: str, args, params, default):
    """
    Choose the parameter value:
    - If the CLI flag appears in sys.argv -> use args.<param_name>
    - Else if param exists in params.yaml -> use params[param_name]
    - Else -> use args.<param_name> (which is set to default by argparse)
    """
    if cli_flag in sys.argv:
        return getattr(args, param_name)
    if param_name in params:
        return params[param_name]
    return getattr(args, param_name)


def load_data(processed_dir):
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    return train, test


def featurize(df, exclude_direct_leakage=True):
    if exclude_direct_leakage:
        feats = [
            'rank_diff', 'tournaments_diff',
            'hist_avg_points_diff', 'hist_std_points_diff',
            'hist_slope_diff', 'hist_count_diff'
        ]
        # Add ratio features if they exist
        if 'points_ratio' in df.columns:
            feats.append('points_ratio')
        if 'rank_ratio' in df.columns:
            feats.append('rank_ratio')
    else:
        feats = [
            'points_diff', 'rank_diff', 'tournaments_diff',
            'hist_avg_points_diff', 'hist_std_points_diff',
            'hist_slope_diff', 'hist_count_diff'
        ]

    available_feats = [f for f in feats if f in df.columns]
    print(f"Using features: {available_feats}")

    X = df[available_feats].fillna(0.0).astype(float)
    y = df['label'].astype(int)
    return X, y


def evaluate_imbalanced_model(y_true, y_pred, y_proba, set_name="Validation"):
    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred)
    # handle case where matrix is not 2x2
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # fallback - fill zeros
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\n{set_name} Results:")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {
        'accuracy': float(accuracy),
        'auc': float(auc) if not np.isnan(auc) else None,
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--mlflow-uri', default='file:./mlruns')  # default to local mlruns
    parser.add_argument('--mlflow-experiment', default='automated_pipeline_experiment')
    parser.add_argument('--n-splits', type=int, default=3)
    parser.add_argument('--exclude-leaky-features', action='store_true', default=True)
    parser.add_argument('--use-smote', action='store_true', default=True)
    parser.add_argument('--calibrate-probabilities', action='store_true', default=True)

    # Hyperparams tunable via params.yaml under "train"
    parser.add_argument('--n-estimators-cv', type=int, default=100)
    parser.add_argument('--n-estimators-final', type=int, default=150)
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)

    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    params = load_params_file(ROOT)

    # determine final effective params (CLI flag wins if present in sys.argv)
    n_splits = int(param_value('--n-splits', 'n_splits', args, params, args.n_splits))
    exclude_leaky = bool(param_value('--exclude-leaky-features', 'exclude_leaky_features', args, params, args.exclude_leaky_features))
    use_smote = bool(param_value('--use-smote', 'use_smote', args, params, args.use_smote))
    calibrate_probs = bool(param_value('--calibrate-probabilities', 'calibrate_probabilities', args, params, args.calibrate_probabilities))

    n_estimators_cv = int(param_value('--n-estimators-cv', 'n_estimators_cv', args, params, args.n_estimators_cv))
    n_estimators_final = int(param_value('--n-estimators-final', 'n_estimators_final', args, params, args.n_estimators_final))
    max_depth = int(param_value('--max-depth', 'max_depth', args, params, args.max_depth))
    learning_rate = float(param_value('--learning-rate', 'learning_rate', args, params, args.learning_rate))
    subsample = float(param_value('--subsample', 'subsample', args, params, args.subsample))
    colsample_bytree = float(param_value('--colsample-bytree', 'colsample_bytree', args, params, args.colsample_bytree))

    mlflow_uri = param_value('--mlflow-uri', 'mlflow_uri', args, params, args.mlflow_uri)
    mlflow_experiment = param_value('--mlflow-experiment', 'mlflow_experiment', args, params, args.mlflow_experiment)

    processed_dir = args.processed_dir
    models_dir = args.models_dir

    os.makedirs(models_dir, exist_ok=True)

    # Simple logging of chosen params
    chosen = {
        "n_splits": n_splits,
        "exclude_leaky_features": exclude_leaky,
        "use_smote": use_smote,
        "calibrate_probabilities": calibrate_probs,
        "n_estimators_cv": n_estimators_cv,
        "n_estimators_final": n_estimators_final,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "mlflow_uri": mlflow_uri,
        "mlflow_experiment": mlflow_experiment
    }
    print("Effective training parameters:", json.dumps(chosen, indent=2))

    # Load data
    train_df, test_df = load_data(processed_dir)

    print(f"Train label distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Test label distribution:\n{test_df['label'].value_counts().sort_index()}")

    # Featurize
    X_train, y_train = featurize(train_df, exclude_direct_leakage=exclude_leaky)
    X_test, y_test = featurize(test_df, exclude_direct_leakage=exclude_leaky)

    # Handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # For scale_pos_weight we want ratio of negative to positive, XGBoost expects sum(negative)/sum(positive)
    # But original script used weight_ratio = class_weights[1] / class_weights[0]
    # Keep weight_ratio for compatibility
    if len(class_weights) >= 2:
        weight_ratio = float(class_weights[1] / class_weights[0])
    else:
        weight_ratio = 1.0

    # -------------------
    # MLflow setup (improved)
    # Priority:
    # 1) MLFLOW_TRACKING_URI env var
    # 2) CLI/params mlflow_uri
    # Also normalizes file: URIs to absolute paths relative to repo root (ROOT)
    # -------------------
    env_mlf = os.environ.get("MLFLOW_TRACKING_URI")
    if env_mlf:
        resolved_mlflow_uri = env_mlf
        print(f"Using MLFLOW_TRACKING_URI from environment: {resolved_mlflow_uri}")
    else:
        resolved_mlflow_uri = mlflow_uri
        print(f"No MLFLOW_TRACKING_URI env var found — using CLI/params value: {resolved_mlflow_uri}")

    # Normalize file: URIs to absolute paths
    try:
        parsed = urlparse(resolved_mlflow_uri)
        if parsed.scheme == "file":
            # Use parsed.path (handles file:./..., file:///abs, file://localhost/abs, etc.)
            file_path = unquote(parsed.path or "")
            if not file_path:
                # fallback if someone passed 'file:' with no path
                file_path = "./mlruns"
            if not os.path.isabs(file_path):
                file_path = os.path.abspath(os.path.join(str(ROOT), file_path))
            else:
                file_path = os.path.abspath(file_path)
            resolved_mlflow_uri = f"file:{file_path}"
            print(f"Normalized local file backend to absolute path: {resolved_mlflow_uri}")
    except Exception as e:
        print("Warning: unable to normalize mlflow URI:", e)

    # Apply tracking URI and try to sanity-check connection
    try:
        mlflow.set_tracking_uri(resolved_mlflow_uri)
        try:
            client = mlflow.tracking.MlflowClient()
            exps = client.list_experiments()
            print(f"Connected to MLflow tracking store. Experiments available: {len(exps)}")
        except Exception as e:
            # Not fatal — may be a file backend or HTTP server not reachable yet
            print("Warning: mlflow client could not list experiments (server may be unreachable or file backend in use):", e)
    except Exception as e:
        print("Failed to set MLflow tracking URI:", e)

    # Ensure experiment exists (create if missing)
    try:
        mlflow.set_experiment(mlflow_experiment)
    except Exception as e:
        print("Warning: could not set/create MLflow experiment:", e)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "exclude_leaky_features": exclude_leaky,
            "use_smote": use_smote,
            "calibrate_probabilities": calibrate_probs,
            "class_weight_ratio": weight_ratio,
            "n_splits": n_splits,
            "n_estimators_cv": n_estimators_cv,
            "n_estimators_final": n_estimators_final,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree
        })

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n=== Fold {fold + 1} ===")

            Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Apply SMOTE if requested
            if use_smote and len(Xtr) > 0:
                # ensure k_neighbors not greater than sample count for minority class
                minority_count = int((ytr == 1).sum())
                k_neighbors = min(5, max(1, minority_count - 1))
                smote = SMOTE(random_state=42 + fold, k_neighbors=k_neighbors)
                Xtr_resampled, ytr_resampled = smote.fit_resample(Xtr, ytr)
                print(f"SMOTE: Resampled from {len(Xtr)} to {len(Xtr_resampled)} samples")
            else:
                Xtr_resampled, ytr_resampled = Xtr, ytr

            # Train model
            clf = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                n_estimators=n_estimators_cv,
                scale_pos_weight=weight_ratio,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42
            )

            # Calibrate if requested - note: calibration wraps estimator
            if calibrate_probs:
                calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3)
                calibrated_clf.fit(Xtr_resampled, ytr_resampled)
                yval_proba = calibrated_clf.predict_proba(Xval)[:, 1]
            else:
                clf.fit(Xtr_resampled, ytr_resampled)
                yval_proba = clf.predict_proba(Xval)[:, 1]

            yval_pred = (yval_proba > 0.5).astype(int)
            fold_metrics = evaluate_imbalanced_model(yval, yval_pred, yval_proba, f"Fold {fold + 1}")
            cv_metrics.append(fold_metrics)

        # Calculate mean CV metrics
        mean_metrics = {}
        for metric in cv_metrics[0].keys():
            values = [m[metric] for m in cv_metrics if (m[metric] is not None)]
            mean_metrics[metric] = float(np.mean(values)) if values else None

        # Log CV metrics to MLflow
        for metric, value in mean_metrics.items():
            if value is not None:
                mlflow.log_metric(f'cv_{metric}', float(value))

        # Train final model on all training data
        print("\n=== Training Final Model ===")

        if use_smote:
            minority_count = int((y_train == 1).sum())
            k_neighbors = min(5, max(1, minority_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"SMOTE final: Resampled from {len(X_train)} to {len(X_train_resampled)} samples")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        final_xgb = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=n_estimators_final,
            scale_pos_weight=weight_ratio,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )

        if calibrate_probs:
            print("Training calibrated model...")
            final_model = CalibratedClassifierCV(final_xgb, method='isotonic', cv=3)
            final_model.fit(X_train_resampled, y_train_resampled)
        else:
            final_model = final_xgb
            final_model.fit(X_train_resampled, y_train_resampled)

        # Test evaluation
        y_test_proba = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba > 0.5).astype(int)

        test_metrics = evaluate_imbalanced_model(y_test, y_test_pred, y_test_proba, "Test")

        # Log test metrics
        for metric, value in test_metrics.items():
            if value is not None:
                mlflow.log_metric(f'test_{metric}', float(value))

        # Save outputs for DVC + downstream usage
        feature_info = {
            'feature_names': X_train.columns.tolist(),
            'exclude_leaky_features': exclude_leaky,
            'is_calibrated': calibrate_probs
        }
        joblib.dump(feature_info, os.path.join(models_dir, 'feature_info.pkl'))

        # Save model(s)
        # Always save a joblib'd version at models/model.pkl for consistency
        model_pickle_path = os.path.join(models_dir, 'model.pkl')
        try:
            joblib.dump(final_model, model_pickle_path)
            print(f"Saved model pickle to {model_pickle_path}")
        except Exception as e:
            print("Warning: failed to joblib.dump the model:", e)

        # For XGBoost underlying model, save booster JSON as well if available
        try:
            if calibrate_probs:
                # calibrated classifier contains calibrated_classifiers_ list with base estimators
                base_estimator = final_model.calibrated_classifiers_[0].estimator
                # save booster JSON
                booster = base_estimator.get_booster()
                xgb_json_path = os.path.join(models_dir, 'xgb_model.json')
                booster.save_model(xgb_json_path)
                mlflow.xgboost.log_model(base_estimator, artifact_path='xgb_model')
                print(f"Saved XGBoost model JSON to {xgb_json_path}")
            else:
                booster = final_model.get_booster()
                xgb_json_path = os.path.join(models_dir, 'xgb_model.json')
                booster.save_model(xgb_json_path)
                mlflow.xgboost.log_model(final_model, artifact_path='xgb_model')
                print(f"Saved XGBoost model JSON to {xgb_json_path}")
        except Exception as e:
            print("Could not save XGBoost booster JSON or log via mlflow.xgboost:", e)

        # If calibrated, also log the calibrated estimator as a sklearn model
        try:
            if calibrate_probs:
                mlflow.sklearn.log_model(final_model, artifact_path='calibrated_xgb_model')
            else:
                # already logged via mlflow.xgboost above
                pass
        except Exception as e:
            print("MLflow model logging error:", e)

        # Save metrics JSON for DVC metrics tracking
        all_metrics = {
            "cv_mean_metrics": mean_metrics,
            "test_metrics": test_metrics,
            "chosen_params": chosen
        }
        metrics_path = os.path.join(models_dir, "metrics.json")
        with open(metrics_path, "w") as mf:
            json.dump(all_metrics, mf, indent=2, default=lambda x: None)

        # Log metrics.json as artifact
        try:
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
        except Exception as e:
            print("Could not log metrics.json to mlflow:", e)

        # Print probability distribution
        print(f"\nProbability distribution on test set:")
        print(f"Min: {y_test_proba.min():.4f}, Max: {y_test_proba.max():.4f}")
        print(f"Mean: {y_test_proba.mean():.4f}, Std: {y_test_proba.std():.4f}")

        # Show sample predictions
        print(f"\nSample predictions (first 10):")
        for i in range(min(10, len(y_test_proba))):
            print(f"True: {y_test.iloc[i]}, Pred: {y_test_pred[i]}, Prob: {y_test_proba[i]:.4f}")



if __name__ == "__main__":
    main()
