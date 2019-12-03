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


run_three_echo_test() {
    #
    # Runs tedana three-echo test
    #

    cprint "RUNNING THREE-ECHO TEST"
    make three-echo
    cprint "THREE-ECHO TEST PASSED !"
}

run_four_echo_test() {
    #
    # Runs tedana four-echo test
    cprint "RUNNING four-ECHO TEST"
    make four-echo
    cprint "FOUR-ECHO TEST PASSED !"
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
    run_three_echo_test
    run_four_echo_test
    run_five_echo_test

    cprint "FINISHED RUNNING ALL TESTS -- GREAT SUCCESS !"
}
