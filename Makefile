.PHONY: setup test clean

setup:
	uv sync

test:
	uv run pytest tests/

clean:
	rm -rf .venv/ build/ dist/ src/pocketmind.egg-info/ .pytest_cache/
