.PHONY: setup lint format typecheck test test-fast bench run docker-build clean doctor

setup:
	python scripts/tasks.py setup

lint:
	python scripts/tasks.py lint

format:
	python scripts/tasks.py format

typecheck:
	python scripts/tasks.py typecheck

test:
	python scripts/tasks.py test

test-fast:
	python scripts/tasks.py test-fast

bench:
	python scripts/tasks.py bench

run:
	python scripts/tasks.py run

docker-build:
	python scripts/tasks.py docker-build

clean:
	python scripts/tasks.py clean

doctor:
	python scripts/tasks.py doctor
