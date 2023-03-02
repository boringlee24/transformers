#!/bin/bash

accelerate launch examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path valhalla/bart-large-finetuned-squadv1 \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu $2
accelerate launch examples/pytorch/question-answering/run_qa_no_trainer.py --model_name_or_path allenai/led-base-16384 \
        --dataset_name squad --max_seq_length 384 --doc_stride 128 --benchmarking \
        --gpu-type $1 --num-gpu $2
   
        
