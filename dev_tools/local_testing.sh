#!/usr/bin/env bash

update_tedana_dockerfile() {
    #
    # Generates Dockerfile to build Docker image for local tedana testing
    #

    get_three_echo_test_data='
        mkdir -p /data/three-echo /data/test/three-echo
        && curl -L -o /data/three-echo/three_echo_Cornell_zcat.nii.gz https://osf.io/8fzse/download
        && curl -L -o TED.Cornell_processed_three_echo_dataset.tar.xz https://osf.io/u65sq/download
        && tar xf TED.Cornell_processed_three_echo_dataset.tar.xz --no-same-owner -C /data/test/three-echo/
        && rm -f TED.Cornell_processed_three_echo_dataset.tar.xz'
    get_five_echo_test_data='
        mkdir /data/five-echo /data/test/five-echo
        && curl -L -o five_echo_NIH.tar.xz https://osf.io/ea5v3/download
        && tar xf five_echo_NIH.tar.xz -C /data/five-echo
        && rm -f five_echo_NIH.tar.xz
        && curl -L -o TED.p06.tar.xz https://osf.io/fr6mx/download
        && tar xf TED.p06.tar.xz --no-same-owner -C /data/test/five-echo/
        && rm -f TED.p06.tar.xz'2
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
      --run "${get_three_echo_test_data}"                                      \
      --run "${get_five_echo_test_data}"                                       \
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

    update_tedana_dockerfile
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


run_integration_tests() {
    #
    # Runs tedana integration tests; passes any parameters to py.test
    #

    cprint "RUNNING INTEGRATION TESTS"
    source activate tedana_py36
    python setup.py -q install
    py.test "$@" tedana/tests/test_integration.py
}


run_unit_tests() {
    #
    # Runs tedana unit tests for Python 3.5, 3.6, and 3.7 environments
    #

    for pyenv in tedana_py35 tedana_py36 tedana_py37; do
        cprint "RUNNING UNIT TESTS FOR PYTHON ENVIRONMENT: ${pyenv}"
        source activate ${pyenv}
        python setup.py -q install
        py.test --ignore-glob=tedana/tests/test_integration.py tedana
    done
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
    run_unit_tests
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
