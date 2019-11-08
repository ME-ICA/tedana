.PHONY: all lint

all_tests: lint unittest integration

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on tedana"
	@echo "  integration	to run integration tests on tedana"
	@echo "  all_tests		to run 'lint', 'unittest', and 'integration'"

lint:
	@flake8 tedana

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=tedana tedana/

integration:
	@py.test --cov-append --cov-report term-missing --cov=tedana tedana/tests/test_integration.py
