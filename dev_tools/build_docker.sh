#!/usr/bin/env bash

update_dev_dockerfile() {
    #
    # Generates Dockerfile to build Docker image for local tedana testing
    #

    generate_ipython_config="
        /opt/conda/envs/tedana_py36/bin/ipython profile create
        && sed -i 's/#c.InteractiveShellApp.extensions = \[\]/c.InteractiveShellApp.extensions = \['\''autoreload'\''\]/g' /root/.ipython/profile_default/ipython_config.py"

    call_dir=${PWD}
    git_root=$( git rev-parse --show-toplevel ); cd "${git_root}"
    docker run --rm kaczmarj/neurodocker:0.6.0 generate docker                 \
      --base debian:latest                                                     \
      --pkg-manager apt                                                        \
      --env LANG=C.UTF-8 LC_ALL=C.UTF-8                                        \
      --install curl git wget bzip2 ca-certificates sed openssh-client         \
      --run "mkdir -p /tedana/dev_tools/envs"                                  \
      --copy ./dev_tools/envs/py35_env.yml /tedana/dev_tools/envs/py35_env.yml \
      --copy ./dev_tools/envs/py36_env.yml /tedana/dev_tools/envs/py36_env.yml \
      --copy ./dev_tools/envs/py37_env.yml /tedana/dev_tools/envs/py37_env.yml \
      --miniconda create_env=tedana_py35                                       \
                  install_path=/opt/conda                                      \
                  yaml_file=/tedana/dev_tools/envs/py35_env.yml                \
                  activate_env=false                                           \
      --miniconda create_env=tedana_py36                                       \
                  install_path=/opt/conda                                      \
                  yaml_file=/tedana/dev_tools/envs/py36_env.yml                \
                  activate_env=true                                            \
      --miniconda create_env=tedana_py37                                       \
                  install_path=/opt/conda                                      \
                  yaml_file=/tedana/dev_tools/envs/py37_env.yml                \
                  activate_env=false                                           \
      --run "${generate_ipython_config}"                                       \
      --copy "./dev_tools" /tedana/dev_tools                                   \
      --add-to-entrypoint "source activate tedana_py36"                        \
      --add-to-entrypoint "source /tedana/dev_tools/run_tests.sh"              \
      --workdir /tedana                                                        \
      > ./Dockerfile_dev
    cd "${call_dir}"
}


build_dev_image() {
    #
    # Recreates local Dockerfile and builds tedana/tedana-dev:local Docker image
    #

    if [ ! -z "${1}" ]; then
        tag="${1}"
    else
        tag=local
    fi

    update_dev_dockerfile
    docker build --tag tedana/tedana-dev:"${tag}" -f Dockerfile_dev .
}
