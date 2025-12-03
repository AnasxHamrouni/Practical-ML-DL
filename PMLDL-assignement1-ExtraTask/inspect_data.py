import pandas as pd
pd.set_option('display.max_columns', 50)

train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")

print("TRAIN shape:", train.shape)
print("TEST  shape:", test.shape)
print("\nLabel distribution (train):")
print(train['label'].value_counts(dropna=False))

print("\nLabel distribution (test):")
print(test['label'].value_counts(dropna=False))

print("\nTrain features summary (numeric):")
print(train[['points_diff','rank_diff','tournaments_diff',
             'hist_avg_points_diff','hist_std_points_diff','hist_slope_diff','hist_count_diff']].describe().T)

print("\nSample rows (train):")
print(train.sample(10, random_state=1).head(10).to_string(index=False))
