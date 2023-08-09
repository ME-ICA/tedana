import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skipintegration", action="store_true", default=False, help="Skip integration tests."
    )


@pytest.fixture
def skip_integration(request):
    return request.config.getoption("--skipintegration")


def three_echo_location():
    return "three_echo_location"


def three_echo_outputs():
    return "three_echo_outputs"


def five_echo_location():
    return "five_echo_location"


def five_echo_outputs():
    return "five_echo_outputs"
