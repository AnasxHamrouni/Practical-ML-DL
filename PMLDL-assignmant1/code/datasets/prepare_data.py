import os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def build_timeseries(raw_dir: str) -> pd.DataFrame:
    types = ['MS', 'WS', 'MD', 'WD', 'XD']
    rows = []
    
    base_path = Path(raw_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    
    dates = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
    # print(f"Found {len(dates)} snapshot dates")
    
    for date in dates:
        folder = base_path / date
        for t in types:
            fname_prefix = f"{t}_{date}"
            for f in folder.iterdir():
                if f.is_file() and (f.name.startswith(fname_prefix) or f.stem == fname_prefix):
                    try:
                        for sep in [',', '\t', ';', ' ']:
                            try:
                                df = pd.read_csv(f, sep=sep)
                                if len(df.columns) > 1: 
                                    break
                            except:
                                continue
                        
                        if df.empty:
                            continue
                            
                        df['snapshot_date'] = date
                        df['match_type'] = t
                        rows.append(df)
                        break
                    except Exception as e:
                        print(f"Warning: Could not read {f}: {e}")
                        continue
    
    if not rows:
        raise RuntimeError('No valid snapshot files found.')
    
    combined = pd.concat(rows, ignore_index=True, sort=False)
    
    # Convert snapshot_date to datetime
    try:
        combined['snapshot_date'] = pd.to_datetime(combined['snapshot_date'])
    except:
        print("Warning: Could not parse snapshot dates")
    
    print(f"Combined data shape: {combined.shape}")
    return combined

def compute_history_features(combined: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    def get_entity_id(row):
        if row['match_type'] in ['MS', 'WS']:
            return str(row.get('player_id', ''))
        else:
            id1 = str(row.get('player_id_one', ''))
            id2 = str(row.get('player_id_two', ''))
            return f"{id1}_{id2}" if id1 and id2 else None
    
    combined = combined.copy()
    combined['entity_id'] = combined.apply(get_entity_id, axis=1)
    combined['points'] = pd.to_numeric(combined.get('points', np.nan), errors='coerce')
    
    # Remove rows without essential data
    mask = (combined['entity_id'].notna()) & (combined['points'].notna())
    combined = combined[mask].copy()
    
    if combined.empty:
        return pd.DataFrame()
    
    # Sort by date for proper time series operations
    combined = combined.sort_values(['entity_id', 'snapshot_date']).reset_index(drop=True)
    
    # Group by entity and compute rolling historical features
    historical_data = []
    
    for (match_type, entity_id), group in combined.groupby(['match_type', 'entity_id']):
        if len(group) < 2:  
            continue
            
        group = group.sort_values('snapshot_date')
        dates = group['snapshot_date'].values
        points = group['points'].values
        
        # For each date, compute features using only prior data
        for i in range(1, len(group)):
            current_date = dates[i]
            prior_data = group.iloc[:i]  # Data strictly before current_date
            
            if len(prior_data) == 0:
                continue
                
            # Use last window points (or all available if less than window)
            window_data = prior_data.tail(window)
            
            # Compute features
            hist_points = window_data['points'].values
            avg_points = np.mean(hist_points) if len(hist_points) > 0 else np.nan
            std_points = np.std(hist_points, ddof=1) if len(hist_points) > 1 else 0.0
            
            # Compute slope using linear regression
            if len(hist_points) >= 2:
                x = np.arange(len(hist_points))
                slope = np.polyfit(x, hist_points, 1)[0]
            else:
                slope = 0.0
                
            count_history = len(prior_data)
            
            historical_data.append({
                'snapshot_date': current_date,
                'match_type': match_type,
                'entity_id': entity_id,
                'hist_avg_points': avg_points,
                'hist_std_points': std_points,
                'hist_slope_points': slope,
                'hist_count': count_history
            })
    
    if not historical_data:
        return pd.DataFrame()
        
    hist_df = pd.DataFrame(historical_data)
    hist_df['snapshot_date'] = pd.to_datetime(hist_df['snapshot_date'])
    
    print(f"Historical features computed for {len(hist_df)} entity-date combinations")
    return hist_df

def create_balanced_pairs(group: pd.DataFrame, max_pairs_per_entity: int = 50) -> pd.DataFrame:
    entities = group[['entity_id', 'points', 'rank', 'number_of_tournaments', 
                     'hist_avg_points', 'hist_std_points', 'hist_slope_points', 'hist_count']].copy()
    
    if len(entities) < 2:
        return pd.DataFrame()
    
    # Sort by points for more meaningful pairings
    entities = entities.sort_values('points', ascending=False).reset_index(drop=True)
    
    pairs = []
    n_entities = len(entities)
    
    # Create pairs with sampling strategy
    for i in range(min(n_entities, max_pairs_per_entity)):
        entity_a = entities.iloc[i]
        
        # Sample opponents from different point ranges
        point_range = entity_a['points'] * 0.5  # ±50% points difference
        lower_bound = entity_a['points'] - point_range
        upper_bound = entity_a['points'] + point_range
        
        potential_opponents = entities[
            (entities.index != i) & 
            (entities['points'] >= lower_bound) & 
            (entities['points'] <= upper_bound)
        ]
        
        if len(potential_opponents) > 10:
            potential_opponents = potential_opponents.sample(10, random_state=42)
        
        for _, entity_b in potential_opponents.iterrows():
            label = 1 if entity_a['points'] > entity_b['points'] else 0
            
            pair_data = {
                'player_a_id': entity_a['entity_id'],
                'player_b_id': entity_b['entity_id'],
                'player_a_points': entity_a['points'],
                'player_b_points': entity_b['points'],
                'player_a_rank': entity_a.get('rank', np.nan),
                'player_b_rank': entity_b.get('rank', np.nan),
                'player_a_num_tournaments': entity_a.get('number_of_tournaments', np.nan),
                'player_b_num_tournaments': entity_b.get('number_of_tournaments', np.nan),
                'player_a_hist_avg_points': entity_a.get('hist_avg_points', np.nan),
                'player_b_hist_avg_points': entity_b.get('hist_avg_points', np.nan),
                'player_a_hist_std_points': entity_a.get('hist_std_points', np.nan),
                'player_b_hist_std_points': entity_b.get('hist_std_points', np.nan),
                'player_a_hist_slope_points': entity_a.get('hist_slope_points', 0.0),
                'player_b_hist_slope_points': entity_b.get('hist_slope_points', 0.0),
                'player_a_hist_count': entity_a.get('hist_count', 0),
                'player_b_hist_count': entity_b.get('hist_count', 0),
                'label': label
            }
            pairs.append(pair_data)
    
    return pd.DataFrame(pairs)

def make_pairwise_examples_with_history(combined: pd.DataFrame, hist: pd.DataFrame, out_path: str):
    # Prepare base data
    combined = combined.copy()
    combined['points'] = pd.to_numeric(combined.get('points', np.nan), errors='coerce')
    
    def get_entity_id(row):
        if row['match_type'] in ['MS', 'WS']:
            return str(row.get('player_id', ''))
        else:
            id1 = str(row.get('player_id_one', ''))
            id2 = str(row.get('player_id_two', ''))
            return f"{id1}_{id2}" if id1 and id2 else None
    
    combined['entity_id'] = combined.apply(get_entity_id, axis=1)
    combined = combined.dropna(subset=['entity_id', 'points'])
    
    # Merge historical features
    if not hist.empty:
        merged = combined.merge(
            hist, 
            on=['snapshot_date', 'match_type', 'entity_id'], 
            how='left',
            suffixes=('', '_hist')
        )
    else:
        merged = combined.copy()
        # Add empty historical columns
        hist_cols = ['hist_avg_points', 'hist_std_points', 'hist_slope_points', 'hist_count']
        for col in hist_cols:
            merged[col] = np.nan
    
    # Fill NaN historical values
    hist_fill_values = {
        'hist_avg_points': 0.0,
        'hist_std_points': 0.0,
        'hist_slope_points': 0.0,
        'hist_count': 0
    }
    
    for col, fill_value in hist_fill_values.items():
        merged[col] = merged[col].fillna(fill_value)
    
    # Create pairwise examples
    all_examples = []
    
    for (date, mt), group in merged.groupby(['snapshot_date', 'match_type']):
        if len(group) < 2:
            continue
            
        pairs = create_balanced_pairs(group)
        if not pairs.empty:
            pairs['snapshot_date'] = str(pd.to_datetime(date).date())
            pairs['match_type'] = mt
            all_examples.append(pairs)
    
    if not all_examples:
        raise ValueError("No pairwise examples could be generated")
    
    X = pd.concat(all_examples, ignore_index=True)
    
    # Feature engineering 
    X['points_diff'] = X['player_a_points'] - X['player_b_points']
    X['rank_diff'] = X['player_b_rank'] - X['player_a_rank']  # Higher rank = lower number
    X['tournaments_diff'] = X['player_a_num_tournaments'] - X['player_b_num_tournaments']
    
    X['hist_avg_points_diff'] = X['player_a_hist_avg_points'] - X['player_b_hist_avg_points']
    X['hist_std_points_diff'] = X['player_a_hist_std_points'] - X['player_b_hist_std_points']
    X['hist_slope_diff'] = X['player_a_hist_slope_points'] - X['player_b_hist_slope_points']
    X['hist_count_diff'] = X['player_a_hist_count'] - X['player_b_hist_count']
    
    # Additional features
    X['points_ratio'] = X['player_a_points'] / (X['player_b_points'] + 1e-8)
    X['rank_ratio'] = (X['player_b_rank'] + 1) / (X['player_a_rank'] + 1e-8)  # Handle division by zero
    
    # Split by time (last date for test)
    dates_sorted = sorted(X['snapshot_date'].unique())
    if len(dates_sorted) >= 2:
        train_dates = dates_sorted[:-1]
        test_dates = [dates_sorted[-1]]
    else:
        train_dates = dates_sorted
        test_dates = dates_sorted
    
    train = X[X['snapshot_date'].isin(train_dates)].reset_index(drop=True)
    test = X[X['snapshot_date'].isin(test_dates)].reset_index(drop=True)
    
    # Analyze label distribution
    train_label_dist = train['label'].value_counts().sort_index()
    test_label_dist = test['label'].value_counts().sort_index()
    
    print(f"\nLabel distribution (train):")
    print(train_label_dist)
    print(f"\nLabel distribution (test):")
    print(test_label_dist)
    
    # Save results
    os.makedirs(out_path, exist_ok=True)
    train.to_csv(os.path.join(out_path, 'train.csv'), index=False)
    test.to_csv(os.path.join(out_path, 'test.csv'), index=False)
    
    print(f"\nSaved train ({len(train):,} rows) and test ({len(test):,} rows) to {out_path}")
    print(f"Train features shape: {train.shape}")
    print(f"Test features shape: {test.shape}")
    
    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='data/raw/out')
    parser.add_argument('--out-dir', default='data/processed')
    parser.add_argument('--window', type=int, default=5, help='History window size')
    parser.add_argument('--max-pairs', type=int, default=50, help='Max pairs per entity')
    args = parser.parse_args()

    print("Loading and processing data...")
    combined = build_timeseries(raw_dir=args.raw_dir)
    hist = compute_history_features(combined, window=args.window)
    train, test = make_pairwise_examples_with_history(combined, hist, args.out_dir)
    
    print("\n=== Data Quality Report ===")
    print(f"Historical features coverage: {hist.shape[0] / combined.shape[0] * 100:.1f}%")
    print(f"Train set balance: {train['label'].mean() * 100:.1f}% positive class")
    print(f"Test set balance: {test['label'].mean() * 100:.1f}% positive class")