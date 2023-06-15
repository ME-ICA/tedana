.PHONY: all lint

all_tests: lint unittest three-echo four-echo five-echo reclassify t2smap

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on tedana"
	@echo "  three-echo	to run the three-echo test set on tedana"
	@echo "  five-echo	to run the five-echo test set on tedana"
	@echo "  t2smap	to run the t2smap integration test set on tedana"
	@echo "  all_tests		to run 'lint', 'unittest', and 'integration'"

lint:
	@black --check --diff tedana
	@flake8 tedana

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=tedana tedana/

three-echo:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=tedana -k test_integration_three_echo tedana/tests/test_integration.py

four-echo:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=tedana -k test_integration_four_echo tedana/tests/test_integration.py

five-echo:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=tedana -k test_integration_five_echo tedana/tests/test_integration.py

reclassify:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=tedana -k test_integration_reclassify tedana/tests/test_integration.py

t2smap:
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=tedana -k test_integration_t2smap tedana/tests/test_integration.py
