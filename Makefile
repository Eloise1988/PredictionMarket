.PHONY: install run run-once dry-run stream stream-dry-run backtest test

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && python3 -m prediction_agent.app

run-once:
	. .venv/bin/activate && python3 -m prediction_agent.app --once

dry-run:
	. .venv/bin/activate && python3 -m prediction_agent.app --once --dry-run

stream:
	. .venv/bin/activate && python3 -m prediction_agent.streaming.service

stream-dry-run:
	. .venv/bin/activate && python3 -m prediction_agent.streaming.service --dry-run

backtest:
	. .venv/bin/activate && python3 -m prediction_agent.backtest.runner --save-report

test:
	. .venv/bin/activate && python3 -m unittest discover -s tests -p 'test_*.py'
