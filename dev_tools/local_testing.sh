#!/usr/bin/env bash

generate_tedana_dockerfile() {
    #
    # Generates Dockerfile to build Docker image for local tedana testing
    #

    get_three_echo_data='
        mkdir -p /data/three-echo
        && curl -L -o /data/three-echo/three_echo_Cornell_zcat.nii.gz https://osf.io/8fzse/download'
    get_five_echo_data='
        mkdir /data/five-echo
        && curl -L -o five_echo_NIH.tar.xz https://osf.io/ea5v3/download
        && tar xf five_echo_NIH.tar.xz -C /data/five-echo
        && rm -f five_echo_NIH.tar.xz'
    get_three_echo_reg='
        mkdir -p /data/test/three-echo
        && curl -L -o TED.Cornell_processed_three_echo_dataset.tar.xz https://osf.io/u65sq/download
        && tar xf TED.Cornell_processed_three_echo_dataset.tar.xz --no-same-owner -C /data/test/three-echo/
        && rm -f TED.Cornell_processed_three_echo_dataset.tar.xz'
    get_five_echo_reg='
        mkdir -p /data/test/five-echo
        && curl -L -o TED.p06.tar.xz https://osf.io/fr6mx/download
        && tar xf TED.p06.tar.xz --no-same-owner -C /data/test/five-echo/
        && rm -f TED.p06.tar.xz'
    generate_ipython_config="
        /opt/conda/envs/tedana_py36/bin/ipython profile create
        && sed -i 's/#c.InteractiveShellApp.extensions = \[\]/c.InteractiveShellApp.extensions = \['\''autoreload'\''\]/g' /root/.ipython/profile_default/ipython_config.py"

    docker run --rm kaczmarj/neurodocker:0.6.0 generate docker                 \
      --base debian:latest                                                     \
      --pkg-manager apt                                                        \
      --env LANG=C.UTF-8 LC_ALL=C.UTF-8                                        \
      --install curl git wget bzip2 ca-certificates sed                        \
      --run "mkdir -p /dev_tools/envs /tedana"                                 \
      --copy ./dev_tools/envs/py35_env.yml /dev_tools/envs/py35_env.yml        \
      --copy ./dev_tools/envs/py36_env.yml /dev_tools/envs/py36_env.yml        \
      --copy ./dev_tools/envs/py37_env.yml /dev_tools/envs/py37_env.yml        \
      --miniconda create_env=tedana_py35                                       \
                  install_path=/opt/conda                                      \
                  yaml_file=/dev_tools/envs/py35_env.yml                       \
                  activate_env=false                                           \
      --miniconda create_env=tedana_py36                                       \
                  install_path=/opt/conda                                      \
                  yaml_file=/dev_tools/envs/py36_env.yml                       \
                  activate_env=true                                            \
      --miniconda create_env=tedana_py37                                       \
                  install_path=/opt/conda                                      \
                  yaml_file=/dev_tools/envs/py37_env.yml                       \
                  activate_env=false                                           \
      --run "${get_three_echo_data}"                                           \
      --run "${get_five_echo_data}"                                            \
      --run "${get_three_echo_reg}"                                            \
      --run "${get_five_echo_reg}"                                             \
      --run "${generate_ipython_config}"                                       \
      --copy "./dev_tools/local_testing.sh" /dev_tools/local_testing.sh        \
      --add-to-entrypoint "source activate tedana_py36"                        \
      --add-to-entrypoint "source /dev_tools/local_testing.sh"                 \
      --workdir /tedana                                                        \
      > ./Dockerfile_dev
}


build_tedana_image() {
    #
    # Recreates local Dockerfile and builds tedana/tedana-dev:local Docker image
    #

    if [ ! -z "${1}" ]; then
        tag="${1}"
    else
        tag=local
    fi

    generate_tedana_dockerfile
    docker build --tag tedana/tedana-dev:${tag} -f Dockerfile_dev .
}


cprint() {
    #
    # Prints all supplied arguments as a green string
    #

    if [[ -t 0 ]]; then
        COLS=$( tput cols )
    else
        COLS=80
    fi

    msg=${*}
    eq=$( python -c "print('=' * ((${COLS} - len('${msg}') - 4) // 2))" )
    python -c "print('\033[1m\033[92m${eq}  ${msg}  ${eq}\033[0m')"
}


_check_tedana_outputs() {
    #
    # Runs tedana unit tests for specified Python version / virtual environment
    #
    # Required argments:
    #   dataset         name of dataset to use for testing. should be one of
    #                   [three-echo, five-echo]

    # confirm specification of three-echo or five-echo input
    if [ -z "${1}" ] || { [ "${1}" != "three-echo" ] && [ "${1}" != "five-echo" ]; }; then
        printf 'Must supply dataset name for checking integration test ' >&2
        printf 'outputs; must be one of [three-echo, five-echo]\n' >&2
        return
    fi

    # find file
    find /data/"${1}"/TED."${1}"/* \
        -exec basename {} \; > /data/"${1}"/TED."${1}"/outputs.out

    # set filenames
    f1=/tedana/.circleci/tedana_outputs.txt
    f2=/data/"${1}"/TED."${1}"/outputs.out

    # sort both files, pipe into grep to check for tedana
    # logfile format; should only see one
    comm -13 <(sort -u $f1) <(sort -u $f2) | grep -E -e '^tedana_[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.txt$'
    numlogs=$(comm -13 <(sort -u $f1) <(sort -u $f2) | grep -E -e '^tedana_[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.txt$' | wc -l)
    if [[ ! $numlogs -eq 1 ]]; then
    printf "Incorrect number of logfiles: %s" $numlogs
    fi

    # verify non-log outputs match exactly
    f3=/data/"${1}"/TED."${1}"/outputs_nolog.txt
    find /data/"${1}"/TED."${1}"/* \
        -exec basename {} \; | grep -v outputs.out | sort > /data/"${1}"/TED."${1}"/outputs.out
    cat $f2 | grep -v -E -e '^tedana_[12][0-9]{3}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}.txt$' > $f3
    diff $f1 $f3
}


_run_integration_test() {
    #
    # Runs tedana integration tests for specified dataset
    #
    # Required argments:
    #   dataset         name of dataset to use for testing. should be one of
    #                   [three-echo, five-echo]

    if [ -z "${1}" ] || { [ "${1}" != "three-echo" ] && [ "${1}" != "five-echo" ]; }; then
        printf 'Must supply dataset name for running integration test; ' >&2
        printf 'must be one of [three-echo, five-echo]\n' >&2
        return
    fi
    ds=${1}
    cprint "RUNNING INTEGRATION TESTS FOR DATASET: ${ds}"
    source activate tedana_py36
    python setup.py -q install
    py.test tedana/tests/test_integration_${ds/-/_}.py
    _check_tedana_outputs "${ds}"
}


run_integration_tests() {
    #
    # Runs tedana integration tests for both three-echo and five-echo datasets
    #

    for ds in three-echo five-echo; do
        _run_integration_test ${ds}
    done
}


run_unit_tests() {
    #
    # Runs tedana unit tests for Python 3.5, 3.6, and 3.7 environments
    #

    for pyenv in tedana_py35 tedana_py36 tedana_py37; do
        cprint "RUNNING UNIT TESTS FOR PYTHON ENVIRONMENT: ${pyenv}"
        source activate ${pyenv}
        python setup.py -q install
        py.test --ignore-glob=tedana/tests/test_integration*.py tedana
    done
}


run_lint_tests() {
    #
    # Lints the tedana codebase
    #

    cprint "LINTING TEDANA CODEBASE"
    source activate tedana_py36
    flake8 tedana
}


run_tests() {
    #
    # Runs tedana test suite EXCLUDING FIVE-ECHO TEST
    #

    run_lint_tests
    run_unit_tests
    _run_integration_test three-echo

    cprint "FINISHED RUNNING TESTS! GREAT SUCCESS"
}


run_all_tests() {
    #
    # Runs entire tedana test suite
    #

    run_lint_tests
    run_unit_tests
    run_integration_tests

    cprint "FINISHED RUNNING ALL TESTS! GREAT SUCCESS"
}
