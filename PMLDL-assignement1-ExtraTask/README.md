# Badminton PMLDL extra task

## About repo:
- this repository is cloned from: https://github.com/AnasxHamrouni/PMLDL-assignmant1
+ added github actions workflow automated pipeline

## Automated pipeline:
the pipele:
- installs python + dependencies 
- install docker + docker  compose
- runs dvc pipeline:
   - runs data preprocessing script (Stage 1: Data Engineering)
   - runs model training script (Stage 2: Model Engineering)
   - builds and runs docker images:(Stage 3: Deployment)
      - API container
      - APP container
      - MLFLOW container

## About model
- the model helps to guess the winner of a badminton match, by choosing match type you may insert players current points, ranking and played tournaments.
Match type labeling:
MS - man's singles
WS - woman's singles
MD - man's doubles
WD - woman's doubles
XD - mixed doubles

This project still needs improvements, in the future:
- improve data preprocessing and balancing
- improve model design and mechanism
- improve UI to be able to select players
- improve prediction model to be able to predict tournament winner's ranking

## Stage 1: Data Engineering
### Input Artifacts: 
- Data was cloned from Github repository: https://github.com/SanderP99/Badminton-Data/tree/main
- I cloned repo and pasted folder 'out' (containing players stats and tournaments) into data/raw
### Output Artifacts: 
- data/processed:
   - test.csv
   - train.csv

## Stage 2: Model Engineering
- Includes difference-based metrics (`rank_diff`, `points_diff`)
- Handling Class Imbalance, computes class weights.
- Model Architecture:
   - Uses **XGBoost (`XGBClassifier`)** with:
      - Tunable `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.
      - `scale_pos_weight` for class imbalance.
- Training Strategy:
   - Stratified K-Fold cross-validation (`n_splits=3` default).
   - CV metrics logged: accuracy, AUC, sensitivity, specificity, precision, F1.
- Evaluation:
   - Metrics evaluated on CV folds and test set.
   - Confusion matrix, probability distributions, and classification metrics computed.
- Hyperparameter Management:
   - Supports `params.yaml` for tuning.
   - CLI flags override YAML values for flexible experimentation.

## Stage 3: Deployment
- **mlflow**: Model tracking & artifact storage (port 5000)  
- **api**: REST API for predictions (port 8000)  
- **app**: Frontend dashboard (Streamlit) connecting to API (port 8500) 
- **run from root: docker-compose -f code/deployment/docker-compose.yml up -d --build**


## Run preprocessing & training locally (optional)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/datasets/prepare_data.py --raw-dir data/raw/out --out-dir data/processed
python code/models/train.py --processed-dir data/processed --models-dir models --mlflow-uri http://localhost:5000