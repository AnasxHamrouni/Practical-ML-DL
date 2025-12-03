FROM python:3.12-slim

COPY ./backend/requirements.txt /backend/
RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

COPY ./backend /backend
COPY ./models/best /models

EXPOSE 8000

CMD ["python3", "/backend/backend.py"]