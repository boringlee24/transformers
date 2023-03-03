#!/bin/bash
set -x

CUDA_VISIBLE_DEVICE=0 accelerate launch run_qa_no_trainer.py --model_name_or_path bert-base-uncased \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu 1 &&

sleep 30        

CUDA_VISIBLE_DEVICE=0 accelerate launch run_qa_no_trainer.py --model_name_or_path roberta-base \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu 1 &&

sleep 30        

CUDA_VISIBLE_DEVICE=0 accelerate launch run_qa_no_trainer.py --model_name_or_path distilbert-base-cased \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu 1 &&

sleep 30        

CUDA_VISIBLE_DEVICE=0 accelerate launch run_qa_no_trainer.py --model_name_or_path valhalla/bart-large-finetuned-squadv1 \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu 1 &&

sleep 30        

CUDA_VISIBLE_DEVICE=0 accelerate launch run_qa_no_trainer.py --model_name_or_path microsoft/mpnet-base \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu 1
   
        
