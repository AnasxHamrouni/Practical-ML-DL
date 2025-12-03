import os
import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from imblearn.over_sampling import SMOTE

def load_data(processed_dir):
    train = pd.read_csv(os.path.join(processed_dir,'train.csv'))
    test = pd.read_csv(os.path.join(processed_dir,'test.csv'))
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
    except:
        auc = np.nan
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\n{set_name} Results:")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--mlflow-uri', default='file:./mlflow')
    parser.add_argument('--n-splits', type=int, default=3)
    parser.add_argument('--exclude-leaky-features', action='store_true', default=True)
    parser.add_argument('--use-smote', action='store_true', default=True)
    parser.add_argument('--calibrate-probabilities', action='store_true', default=True)
    args = parser.parse_args()

    # Load data
    train_df, test_df = load_data(args.processed_dir)
    
    print(f"Train label distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Test label distribution:\n{test_df['label'].value_counts().sort_index()}")
    
    # Featurize
    X_train, y_train = featurize(train_df, exclude_direct_leakage=args.exclude_leaky_features)
    X_test, y_test = featurize(test_df, exclude_direct_leakage=args.exclude_leaky_features)
    
    # Handle class imbalance
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    weight_ratio = class_weights[1] / class_weights[0]
    
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("exclude_leaky_features", args.exclude_leaky_features)
        mlflow.log_param("use_smote", args.use_smote)
        mlflow.log_param("calibrate_probabilities", args.calibrate_probabilities)
        mlflow.log_param("class_weight_ratio", weight_ratio)
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\n=== Fold {fold + 1} ===")
            
            Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Apply SMOTE if requested
            if args.use_smote and len(Xtr) > 0:
                smote = SMOTE(random_state=42 + fold, k_neighbors=min(5, sum(ytr == 1)))
                Xtr_resampled, ytr_resampled = smote.fit_resample(Xtr, ytr)
                print(f"SMOTE: Resampled from {len(Xtr)} to {len(Xtr_resampled)} samples")
            else:
                Xtr_resampled, ytr_resampled = Xtr, ytr
            
            # Train model
            clf = xgb.XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                n_estimators=100,
                scale_pos_weight=weight_ratio,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Calibrate if requested
            if args.calibrate_probabilities:
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
            values = [m[metric] for m in cv_metrics if not np.isnan(m[metric])]
            mean_metrics[metric] = np.mean(values) if values else np.nan
        
        # Log CV metrics
        for metric, value in mean_metrics.items():
            if not np.isnan(value):
                mlflow.log_metric(f'cv_{metric}', float(value))
        
        # Train final model on all training data
        print("\n=== Training Final Model ===")
        
        if args.use_smote:
            smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train == 1)))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_train_resampled, y_train_resampled = X_train, y_train
        
        final_xgb = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=150,
            scale_pos_weight=weight_ratio,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train final model with or without calibration
        if args.calibrate_probabilities:
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
            if not np.isnan(value):
                mlflow.log_metric(f'test_{metric}', float(value))
        
        # Save model with proper handling for calibrated vs regular models
        os.makedirs(args.models_dir, exist_ok=True)
        
        # Save feature info
        feature_info = {
            'feature_names': X_train.columns.tolist(),
            'exclude_leaky_features': args.exclude_leaky_features,
            'is_calibrated': args.calibrate_probabilities
        }
        joblib.dump(feature_info, os.path.join(args.models_dir, 'feature_info.pkl'))
        
        # Save the appropriate model type
        if args.calibrate_probabilities:
            # Save calibrated model as pickle
            calibrated_model_path = os.path.join(args.models_dir, 'calibrated_xgb_model.pkl')
            joblib.dump(final_model, calibrated_model_path)
            
            # Also save the underlying XGBoost model for reference
            xgb_model_path = os.path.join(args.models_dir, 'xgb_model.json')
            base_estimator = final_model.calibrated_classifiers_[0].estimator
            base_estimator.get_booster().save_model(xgb_model_path)
            
            mlflow.sklearn.log_model(final_model, artifact_path='calibrated_xgb_model')
            print(f"Saved calibrated model to {calibrated_model_path}")
        else:
            # Save regular XGBoost model
            model_path = os.path.join(args.models_dir, 'xgb_model.json')
            final_model.get_booster().save_model(model_path)
            mlflow.xgboost.log_model(final_model, artifact_path='xgb_model')
            print(f"Saved model to {model_path}")
        
        # Print probability distribution
        print(f"\nProbability distribution on test set:")
        print(f"Min: {y_test_proba.min():.4f}, Max: {y_test_proba.max():.4f}")
        print(f"Mean: {y_test_proba.mean():.4f}, Std: {y_test_proba.std():.4f}")
        
        # Show sample predictions
        print(f"\nSample predictions (first 10):")
        for i in range(min(10, len(y_test_proba))):
            print(f"True: {y_test.iloc[i]}, Pred: {y_test_pred[i]}, Prob: {y_test_proba[i]:.4f}")