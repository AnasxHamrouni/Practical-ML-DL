FROM python:3.12-slim

COPY ./frontend/requirements.txt /frontend/requirements.txt
RUN pip install -r /frontend/requirements.txt 

COPY ./frontend /frontend
CMD ["streamlit", "run", "/frontend/frontend.py"]