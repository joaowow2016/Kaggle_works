.PHONY: setup data train predict test lint typecheck check serve docker all clean
CONFIG ?= configs/default.yaml

setup:
	python -m pip install -r requirements.lock
	python -m pip install --no-deps -e .

data:
	pagrus-age --config $(CONFIG) download

train: data
	pagrus-age --config $(CONFIG) train

predict:
	pagrus-age --config $(CONFIG) predict --input data/raw/seanoe_76263.csv \
	  --output data/predictions/unaged_fish.csv --unaged-only

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

check: lint typecheck test

serve:
	pagrus-age --config $(CONFIG) serve

docker:
	docker build -t pagrus-age .

all: check train predict

clean:
	rm -rf data/processed data/predictions reports models .pytest_cache .mypy_cache .ruff_cache .coverage
