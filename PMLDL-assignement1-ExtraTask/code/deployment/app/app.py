import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL', 'http://api:8000/predict')

st.set_page_config(page_title="Badminton Winner Predictor")
st.title("Badminton match winner predictor")

match_type = st.selectbox('Match type', ['MS','WS','MD','WD','XD'])
col1, col2 = st.columns(2)
with col1:
    player_a_points = st.number_input('Player A points', value=10000.0)
    player_a_rank   = st.number_input('Player A rank', value=1)
    player_a_tourn  = st.number_input('Player A # tournaments', value=10)
with col2:
    player_b_points = st.number_input('Player B points', value=9000.0)
    player_b_rank   = st.number_input('Player B rank', value=2)
    player_b_tourn  = st.number_input('Player B # tournaments', value=8)

if st.button('Predict'):
    payload = {
        'match_type': match_type,
        'player_a_points': float(player_a_points),
        'player_b_points': float(player_b_points),
        'player_a_rank': float(player_a_rank),
        'player_b_rank': float(player_b_rank),
        'player_a_num_tournaments': float(player_a_tourn),
        'player_b_num_tournaments': float(player_b_tourn),
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=5)
        data = res.json()
        st.success(f"Predicted winner: {data['winner']} (prob={data['probability']:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
