train:
	python -m src.train

api:
	uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 1

test-validation:
	pytest -q tests/test_data_validation.py

test-model:
	pytest -q tests/test_model.py

test-api:
	pytest -q tests/test_api.py