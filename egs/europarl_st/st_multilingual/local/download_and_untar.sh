#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1

remove_archive=false

if [ "$1" == --remove-archive ]; then
    remove_archive=true
    shift
fi

data=$1

if [ ! -d "${data}" ]; then
    echo "$0: no such directory ${data}"
    exit 1;
fi

tar_name=v1.1
extension=tar.gz
url=https://www.mllp.upv.es/europarl-st/${tar_name}.${extension}

if [ -f ${data}/${tar_name}.${extension} ]; then
    echo "${data}/${tar_name}.${extension} exists."
else
    wget ${url} -P ${data}
fi

if [ -d ${data}/${tar_name} ]; then
    echo "$0: data was already successfully extracted, nothing to do."
else
    echo "Extracting folder ..."
    tar -zxvf ${data}/${tar_name}.${extension} -C ${data}
fi

if $remove_archive; then
    echo "$0: removing ${data}/${tar_name}.${extension} file since --remove-archive option was supplied."
    rm ${data}/${tar_name}.${extension}
fi
