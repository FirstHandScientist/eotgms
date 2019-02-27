#!/bin/bash
python eotgan_main.py \
--dataroot '../data/mnist' \
--dataset 'mnist' \
--sinkgpu \
--regL 5 \
--lrG 1e-4 \
--lrS 1e-4 \
--emb_net 'EmbeddingNet' \
--margin 5 \
--imageSize 32 \
--batchSize 128 \
--experiment 'result/mnist_eotgan_sfn2' \
--nz 100 \
--sfn 2 \
--nc 1 \
--cuda \
--niter 50 \
--ngpu 1 \
--netS 'triplet'
