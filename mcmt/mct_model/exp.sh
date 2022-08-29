#!/bin/bash

filename="baseline+RW+cam+rankedm0.8a1.4_r0.5c1_newreid"
weight="/home/ykwei/MTMC-SU/mcmt/mct_model/checkpoints/${filename}_10.pth"
expfilename="baseline+RW+cam+rankedm0.8a1.4_r0.5c1_th_exp_cosine"

for i in `seq 0.8 0.05 0.95`;
do
    cd /home/ykwei/MCMT-SU/mcmt/pipeline
    python step4_mutli_camera_tracking.py -w "$weight" -s $i > /dev/null 2>&1
    python step5_merge_results.py
    cd /home/ykwei/MCMT-SU/mcmt/eval
    mv /home/ykwei/MCMT-SU/mcmt/dataset/validation/track3.txt .
    echo Theshold "$i": | tee -a /home/ykwei/MCMT-SU/mcmt/mct_model/logs/$expfilename.txt
    python eval.py ground_truth_validation.txt track3.txt --mct --dstype validation | tee -a /home/ykwei/MCMT-SU/mcmt/mct_model/logs/$expfilename.txt
done