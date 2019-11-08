#!/usr/bin/env bash

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
    # Runs tedana integration tests
    #

    cprint "RUNNING INTEGRATION TESTS"
    make integration
    cprint "INTEGRATION TESTS PASSED !"
}


run_unit_tests() {
    #
    # Runs tedana unit tests
    #

    cprint "RUNNING UNIT TESTS"
    make unittest
    cprint "UNIT TESTS PASSED !"
}


run_lint_tests() {
    #
    # Lints the tedana codebase
    #

    cprint "RUNNING FLAKE8 TO LINT CODEBASE"
    make lint
    cprint "CODEBASE LINTED SUCCESSFULLY !"
}


run_all_tests() {
    #
    # Runs tedana test suite
    #

    run_lint_tests
    run_unit_tests
    run_integration_tests

    cprint "FINISHED RUNNING ALL TESTS -- GREAT SUCCESS !"
}
