<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Huggingface Transformer Benchmarks

This repository is a fork from the [huggingface transformer repository](https://github.com/huggingface/transformers). 

The benchmark is used for our SC'23 paper ``Toward Sustainable HPC: Carbon Footprint Estimation and Environmental Implications of HPC Systems``. Please refer to https://github.com/boringlee24/sc23-sustainability for more information.

## Dependencies

Please refer to the original [huggingface transformer repository](https://github.com/huggingface/transformers) for the dependencies.

## Benchmark Scripts and Data

We use the pytorch question-answering benchmark ``examples/pytorch/question-answering/run_qa_no_trainer.py``

We have collected the performance and operational carbon footprint data from running these benchmarks. The data is available at ``examples/pytorch/question-answering/benchmark_logs``. For example, ``4xv100`` represents running over 4 V100 GPUs, ``carbon_{testcase}.json`` reports the operational carbon, while ``time_{testcase}.json`` reports the mini-batch time, representing performance..

## Running in Container and Cluster

Refer to https://github.com/boringlee24/containerized_distributed_training for the options passed into the python script

