#!/bin/bash

filename="sprcnn-su+RW+cam+rankedm0.8a1.4_r0.5c1_EPFL"
weight="/home/ykwei/MTMC-SU/EPFL/mct_model/checkpoints/${filename}_10.pth"

if [[ ! -e /home/ykwei/MCMT-SU/EPFL/mct_model/logs ]]; then
    mkdir /home/ykwei/MCMT-SU/EPFL/mct_model/logs
fi

rm -fr /home/ykwei/MCMT-SU/EPFL/mct_model/logs/$filename.txt
rm -fr /home/ykwei/MCMT-SU/EPFL/mct_model/logs/$filename\_train.txt

python train.py -o "$filename" | tee -a /home/ykwei/MCMT-SU/EPFL/mct_model/logs/$filename\_train.txt