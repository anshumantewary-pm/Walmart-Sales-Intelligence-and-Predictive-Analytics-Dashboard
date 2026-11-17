.PHONY: install sample pipeline dashboard test lint docker

install:
	pip install -r requirements-dev.txt
	pre-commit install || true

sample:
	python -m src.pipeline --sample

pipeline:
	python -m src.pipeline

dashboard:
	streamlit run src/dashboard/app.py

test:
	pytest -q

lint:
	ruff check .

docker:
	docker build -t walmart-sales-dashboard .
	docker run --rm -p 8501:8501 -v $(PWD)/data/raw:/app/data/raw walmart-sales-dashboard
