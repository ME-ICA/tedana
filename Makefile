.PHONY: all lint

all_tests: lint unittest three-echo five-echo

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on tedana"
	@echo "  three-echo	to run the three-echo test set on tedana"
	@echo "  five-echo	to run the five-echo test set on tedana"
	@echo "  all_tests		to run 'lint', 'unittest', and 'integration'"

lint:
	@flake8 tedana

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=tedana tedana/

three-echo:
	@py.test --cov-append --cov-report term-missing --cov=tedana -k test_integration_three_echo tedana/tests/test_integration.py

four-echo:
	@py.test --cov-append --cov-report term-missing --cov=tedana -k test_integration_four_echo tedana/tests/test_integration.py

five-echo:
	@py.test --cov-append --cov-report term-missing --cov=tedana -k test_integration_five_echo tedana/tests/test_integration.py

