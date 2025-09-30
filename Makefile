PYTHON?=python3
PIP?=$(PYTHON) -m pip
VENV?=.venv

.PHONY: venv install test run app lint

venv:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip

install:
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt

lint:
	. $(VENV)/bin/activate && $(PYTHON) -m ruff check src tests

format:
	. $(VENV)/bin/activate && $(PYTHON) -m ruff format src tests

test:
	. $(VENV)/bin/activate && $(PYTHON) -m pytest -q

run:
	. $(VENV)/bin/activate && $(PYTHON) -m cwa.cli all --user $$CHESSCOM_USERNAME --max-months 2 --engine-depth 12

app:
	. $(VENV)/bin/activate && streamlit run src/cwa/app_streamlit.py
