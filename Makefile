# Variables
VENV_NAME := venv

# Main targets
.PHONY: setup help


help:
	@echo "Available commands:"
	@echo "  make setup    - Create virtual environment and install dependencies"
	@echo "  make help     - Show this help message"

# Setup virtual environment and install dependencies
setup:
	python3 -m venv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install -r requirements.txt
	mkdir -p input

