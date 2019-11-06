import pytest


def pytest_addoption(parser):
    parser.addoption('--skipintegration', action='store_true',
                     default=False, help='Skip integration tests.')
    parser.addoption('--include-five-echo', action='store_true',
                     default=False, help='Run the five-echo test set.')

@pytest.fixture
def skip_integration(request):
    return request.config.getoption('--skipintegration')


@pytest.fixture
def include_five_echo(request):
    return request.config.getoption('--include-five-echo')


def three_echo_location():
    return 'three_echo_location'


def three_echo_outputs():
    return 'three_echo_outputs'


def five_echo_location():
    return 'five_echo_location'


def five_echo_outputs():
    return 'five_echo_outputs'
