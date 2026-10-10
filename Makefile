.PHONY: setup lint format typecheck test test-fast bench run docker-build clean doctor up down demo rollback

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

.PHONY: train
train:
	python scripts/tasks.py train

up:
	python scripts/tasks.py up

down:
	python scripts/tasks.py down

demo:
	python scripts/tasks.py demo

rollback:
	MLP_ROLLBACK_VERSION=$(VERSION) python scripts/tasks.py rollback
