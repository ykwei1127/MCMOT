#!/bin/bash

filename="sprcnn-su+RW+cam+rankedm0.8a1.4_r0.5c1_campus_v3"
weight="/home/ykwei/MTMC-SU/campus/mct_model/checkpoints/${filename}_10.pth"
# expfilename="baseline+RW+cam+rankedm0.8a1.4_r0.5c1_flip128_exp"

if [[ ! -e /home/ykwei/MCMT-SU/campus/mct_model/logs ]]; then
    mkdir /home/ykwei/MCMT-SU/campus/mct_model/logs
fi

rm -fr /home/ykwei/MCMT-SU/campus/mct_model/logs/$filename.txt
rm -fr /home/ykwei/MCMT-SU/campus/mct_model/logs/$filename\_train.txt
# rm -fr /home/ykwei/MCMT-SU/campus/mct_model/logs/$expfilename.txt

python train.py -o "$filename" | tee -a /home/ykwei/MCMT-SU/campus/mct_model/logs/$filename\_train.txt

# for i in `seq 0.8 0.05 0.95`;
# do  
#     cd /home/ykwei/MCMT-SU/campus/pipeline
#     python step4_mutli_camera_tracking.py -w "$weight" -s $i > /dev/null 2>&1
#     python step5_merge_results.py
#     cd /home/ykwei/MCMT-SU/campus/eval
#     mv /home/ykwei/MCMT-SU/campus/dataset/validation/track3.txt .
#     echo Theshold "$i": | tee -a /home/ykwei/MCMT-SU/campus/mct_model/logs/$expfilename.txt
#     python eval.py ground_truth_validation.txt track3.txt --mct --dstype validation | tee -a /home/ykwei/MCMT-SU/campus/mct_model/logs/$expfilename.txt
# done