#!/usr/bin/env bash

get_three_echo_test_data() {
    mkdir -p /tmp/data/three-echo
    curl -s -L -o /tmp/data/three-echo/three_echo_Cornell_zcat.nii.gz https://osf.io/8fzse/download
}
    # && curl -L -o TED.Cornell_processed_three_echo_dataset.tar.xz https://osf.io/u65sq/download
    # && tar xf TED.Cornell_processed_three_echo_dataset.tar.xz --no-same-owner -C /data/test/three-echo/
    # && rm -f TED.Cornell_processed_three_echo_dataset.tar.xz'


get_five_echo_test_data() {
    mkdir -p /tmp/data/five-echo
    curl -s -L https://osf.io/ea5v3/download | tar xJf - -C /tmp/data/five-echo
}
    # && curl -L -o TED.p06.tar.xz https://osf.io/fr6mx/download
    # && tar xf TED.p06.tar.xz --no-same-owner -C /data/test/five-echo/
    # && rm -f TED.p06.tar.xz'


download_data() {
    for ds in three-echo five-echo; do
        datadir=/tmp/data/${ds}
        # if the data directory doesn't exist OR the data directory is empty
        # then download the data for the specified dataset
        if [ ! -d ${datadir} ] || [ ! "$( ls -A ${datadir} )" ]; then
            printf "Downloading ${ds} data into ${datadir}\n"
            get_${ds/-/_}_test_data
        fi
    done
}
