.PHONY: setup run test figures demo docker-build docker-run clean

setup:
	pip install -r requirements.txt

run:
	python experiments/run_all.py

test:
	python -m pytest tests/ -q

figures:
	python assets/generate_figures.py

demo:
	./scripts/demo.sh

docker-build:
	docker compose build

docker-run:
	docker compose up --build

clean:
	rm -f results/*.json results/*.csv
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
