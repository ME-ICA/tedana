#!/usr/bin/env bash

DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIRECTORY}/get_data.sh

cprint() {
    #
    # Prints all supplied arguments as a bold, green string
    #

    if [[ -t 0 ]] && [ ! -z "${TERM}" ]; then
        COLS=$( tput -T screen cols )
    else
        COLS=80
    fi

    msg="${*}"
    eq=$( python -c "print('=' * ((${COLS} - len('${msg}') - 4) // 2))" )
    python -c "print('\033[1m\033[92m${eq}  ${msg}  ${eq}\033[0m')"
}


run_integration_tests() {
    #
    # Runs tedana integration tests; passes any parameters to py.test. If you
    # want to run the five-echo test you need to pass "--include-five-echo"
    #

    download_data
    cprint "RUNNING INTEGRATION TESTS"
    source activate tedana_py36
    py.test "$@" --cov-report term-missing --cov=tedana tedana/tests/test_integration.py
}


run_unit_test() {
    #
    # Runs tedana unit tests for provided Python environment
    #
    # Required argments:
    #   pyenv           name of python environment to use for testing. should
    #                   be one of [tedana_py35, tedana_py36, tedana_py37]
    #

    if [ -z "${1}" ] || { [ "${1}" != "tedana_py35" ] \
                            || [ "${1}" != "tedana_py36" ] \
                            || [ "${1}" != "tedana_py37"]; }; then
        printf 'Must supply python environment name for running unit ' >&2
        printf 'tests. Should be one of [tedana_py35, tedana_py36, ' >&2
        printf 'tedana_py37].\n' >&2
        return
    fi

    cprint "RUNNING UNIT TESTS FOR PYTHON ENVIRONMENT: ${1}"
    source activate "${1}"
    py.test --skipintegration --cov-append --cov-report term-missing --cov=tedana tedana
}


run_lint_tests() {
    #
    # Lints the tedana codebase
    #

    cprint "RUNNING FLAKE8 ON TEDANA DIRECTORY"
    source activate tedana_py36
    flake8 tedana
}


run_tests() {
    #
    # Runs tedana test suite excluding five-echo test by default
    #

    if [ ! -z "${1}" ] && [ "${1}" == "--include-five-echo" ]; then
        run_five_echo='--include-five-echo'
    fi

    run_lint_tests
    for pyenv in tedana_py35 tedana_py36 tedana_py37; do
        run_unit_test "${pyenv}"
    done
    run_integration_tests "${run_five_echo}"

    if [ -z "${run_five_echo}" ]; then
        cprint "FINISHED RUNNING TESTS! GREAT SUCCESS"
    fi
}


run_all_tests() {
    #
    # Runs entire tedana test suite
    #

    run_tests --include-five-echo

    cprint "FINISHED RUNNING ALL TESTS! GREAT SUCCESS"
}
