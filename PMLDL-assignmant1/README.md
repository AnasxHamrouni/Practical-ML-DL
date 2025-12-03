# Badminton ML deployment - quick start

## About
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
- improve UI to be able to select players and predict doubles games
- improve prediction model to be able to predict tournament winner's ranking

## About data:
- Data was cloned from Github repository: https://github.com/SanderP99/Badminton-Data/tree/main
- I cloned repo and pasted folder 'out' (containing players stats and tournaments) into data/raw

## Prerequisites
- Docker & docker-compose
- Python 3.10 for running scripts locally

## Quick local run (fastest)
1. Build & run services (mlflow, api, app):
   cd code/deployment
   docker-compose up --build

2. Open:
   - Streamlit app: http://localhost:8501
   - FastAPI docs: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000

## Run preprocessing & training locally (optional)
install requirements.txt (located in root directory)
delete data/processed
run code/datasets/prepare_data.py
run code/models/train.py

## Run docker compose from root directory
build and run docker compose: docker compose -f code/deployment/docker-compose.yml up --build -d
