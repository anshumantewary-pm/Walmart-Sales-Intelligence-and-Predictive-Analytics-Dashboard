FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data ./data

# Build outputs at startup: real data if present in data/raw, else synthetic demo.
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["sh", "-c", "if [ -f data/raw/train.csv ]; then python -m src.pipeline; else python -m src.pipeline --sample; fi && streamlit run src/dashboard/app.py --server.address=0.0.0.0 --server.port=8501"]
