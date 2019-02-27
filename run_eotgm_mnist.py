#!/bin/bash
python eotgm_main.py \
--dataroot '../data/mnist' \
--dataset 'mnist' \
--sinkgpu \
--regL 15 \
--imageSize 32 \
--batchSize 128 \
--experiment 'result/mnist_eotgm' \
--nz 100 \
--nc 1 \
--cuda \
--niter 50 \
--ngpu 1 \
