.PHONY: help install install-dev test lint format clean docker-build docker-run train eval

PYTHON := python
PIP := pip
REQUIREMENTS := requirements.txt

help:
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make clean          - Clean artifacts"
	@echo "  make train          - Train prototypes"
	@echo "  make eval           - Evaluate prototypes"

# Installation

install:
	${PIP} install -r $(REQUIREMENTS)
	$(PIP) install -e .


# Clean

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

# Training

train:
	$(PYTHON) -m scripts.train_prototypes

eval:
	$(PYTHON) -m scripts.eval_prototypes