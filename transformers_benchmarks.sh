#!/bin/bash

accelerate launch examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path xlnet-base-uncased \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu $2
accelerate launch examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path xlm-mlm-en-2048 \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu $2
accelerate launch examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path roberta-base \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu $2
accelerate launch examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path distilbert-base-cased \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu $2       
        
