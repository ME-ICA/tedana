#!/usr/bin/env bash

<<<<<<< HEAD
DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source ${DIRECTORY}/get_data.sh

=======
>>>>>>> tsala/modularize-metrics
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


<<<<<<< HEAD
run_integration_tests() {
    #
    # Runs tedana integration tests; passes any parameters to py.test
    #

    download_data
    cprint "RUNNING INTEGRATION TESTS"
    source activate tedana_py36
    py.test "$@" --cov-append --cov-report term-missing --cov=tedana tedana/tests/test_integration.py
}


run_unit_test() {
    #
    # Runs tedana unit tests for provided Python environment
    #
    # Required argments:
    #   pyenv           name of python environment to use for testing. should
    #                   be one of [tedana_py35, tedana_py36, tedana_py37]
    #

    cprint "RUNNING UNIT TESTS FOR PYTHON ENVIRONMENT: ${1}"
    source activate "${1}"
    py.test --skipintegration --cov-append --cov-report term-missing --cov=tedana tedana
=======
run_three_echo_test() {
    #
    # Runs tedana three-echo test
    #

    cprint "RUNNING THREE-ECHO TEST"
    make three-echo
    cprint "THREE-ECHO TEST PASSED !"
}

run_five_echo_test() {
    #
    # Runs tedana five-echo test
    cprint "RUNNING FIVE-ECHO TEST"
    make five-echo
    cprint "FIVE-ECHO TEST PASSED !"
}


run_unit_tests() {
    #
    # Runs tedana unit tests
    #

    cprint "RUNNING UNIT TESTS"
    make unittest
    cprint "UNIT TESTS PASSED !"
>>>>>>> tsala/modularize-metrics
}


run_lint_tests() {
    #
    # Lints the tedana codebase
    #

<<<<<<< HEAD
    cprint "RUNNING FLAKE8 ON TEDANA DIRECTORY"
    source activate tedana_py36
    flake8 tedana
=======
    cprint "RUNNING FLAKE8 TO LINT CODEBASE"
    make lint
    cprint "CODEBASE LINTED SUCCESSFULLY !"
>>>>>>> tsala/modularize-metrics
}


run_all_tests() {
    #
<<<<<<< HEAD
    # Runs tedana test suite excluding five-echo test by default
    #

    run_lint_tests
    for pyenv in tedana_py3{5,6,7}; do
        run_unit_test "${pyenv}"
    done
    run_integration_tests

    cprint "FINISHED RUNNING ALL TESTS! GREAT SUCCESS"
=======
    # Runs tedana test suite
    #

    run_lint_tests
    run_unit_tests
    run_three_echo_test
    run_five_echo_test

    cprint "FINISHED RUNNING ALL TESTS -- GREAT SUCCESS !"
>>>>>>> tsala/modularize-metrics
}
