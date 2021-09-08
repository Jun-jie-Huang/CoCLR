# CoSQA and CoCLR for Code Search and Question Answering

This repository contains the data and source code used in ACL 2021 main conference paper  [CoSQA: 20,000+ Web Queries for Code Search and Question Answering](https://aclanthology.org/2021.acl-long.442.pdf).  The CoSQA dataset includes 20,604 human annotated labels for pairs of natural language web queries and codes. The source code contains baseline methods and proposed contrastive learning method dubbed CoCLR to enhance query-code matching. The dataset and source code are created by Beihang University, MSRA NLC group and STCA NLP group. Our codes follow MIT License and our datasets follow Computational Use of Data Agreement (CUDA) License. 

## Repository Structure

- data/qa: this folder contains the query/code pairs for training, dev and test data. For better usage, we copy the CoSQA dataset and WebQueryTest from [CodeXGLUE -- Code Search (WebQueryTest)](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery).
- data/retrieval: this folder contains the data for training, validating and testing a code retriever. The code to obtain data for ablation study in the [paper](https://aclanthology.org/2021.acl-long.442.pdf) is also included.
- code_qa: this folder contains the source code to run code question answering task.
- code_search: this folder contains the source code to run code search task.

## Requirements

```
torch==1.4.0
transformers==2.5.0
tqdm
scikit-learn
nltk
```

## Code Question Answering

### Vanilla Model

#### Step 1: download the checkpoint trained on CodeSearchNet

Please to the first point in [Model Checkpoint section](#model-checkpoint)

#### Step 2: training

```
model=./model/qa_codebert
CUDA_VISIBLE_DEVICES="0" python ./code_qa/run_siamese_test.py \
		--model_type roberta 
		--augment 
		--do_train 
		--do_eval 
		--eval_all_checkpoints 
		--data_dir ./data/qa/ \	
		--train_data_file cosqa-train.json 
		--eval_data_file cosqa-dev.json 
		--max_seq_length 200 
		--per_gpu_train_batch_size 32 
		--per_gpu_eval_batch_size 16 
		--learning_rate 5e-6 
		--num_train_epochs 10 
		--gradient_accumulation_steps 1 
		--evaluate_during_training \
		--warmup_steps 500 \
		--checkpoint_path ./model/codesearchnet-checkpoint \
		--output_dir ${model} \
		--encoder_name_or_path microsoft/codebert-base \
        2>&1 | tee ./qa-train-codebert.log
```

#### Step 3: evaluate on CodeXGLUE - code search (WebQueryTest)

To evaluate on CodeXGLUE WebQueryTest, you can first download the test file from [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) and move the file to `data` directory with the following commands.

```
git clone https://github.com/microsoft/CodeXGLUE
cp CodeXGLUE/Text-Code/NL-code-search-WebQuery/data/test_webquery.json ./data/qa/
```

Then you can evaluate you model and submit the `--test_predictions_output` to CodeXGLUE challenge for the results on the test set.

You can submit the `--test_predictions_output` to CodeXGLUE challenge for the results on the test set.
```
model=./model/qa_codebert
CUDA_VISIBLE_DEVICES="0" python ./code_qa/run_siamese_test.py \
		--model_type roberta  \
		--augment \
		--do_test \
		--data_dir ./data/qa \
		--test_data_file test_webquery.json \
		--max_seq_length 200 \
		--per_gpu_eval_batch_size 2 \
		--output_dir ${model}/checkpoint-best-aver/ \
		--encoder_name_or_path microsoft/codebert-base \
		--pred_model_dir ${model}/checkpoint-best-aver/ \
		--test_predictions_output ${model}/webquery_predictions.txt \
		2>&1| tee ./qa-test-codebert.log
```

### CoCLR on Code Question Answering

To apply CoCLR on the task of code question answering, you can run the commands with the following steps.

#### Step 1: download the checkpoint trained on CodeSearchNet

Please to the first point in [Model Checkpoint section](#model-checkpoint)

#### Step 2: create query-rewritten data

```
cd data
python augment_qra.py --task qa --qra_mode delete
python augment_qra.py --task qa --qra_mode copy
python augment_qra.py --task qa --qra_mode switch
cd ../
```

#### Step 3: training

```
qra=switch
model=./model/qa_codebert_${qra}
CUDA_VISIBLE_DEVICES="0" python ./code_qa/run_siamese_test.py \
		--model_type roberta \
		--augment \
		--do_train \
		--do_eval \
		--eval_all_checkpoints \
		--data_dir ./data/qa/ \
		--train_data_file cosqa-train-qra-${qra}-29707.json \
		--eval_data_file cosqa-dev.json \
		--max_seq_length 200 \
		--per_gpu_train_batch_size 32 \
		--per_gpu_eval_batch_size 16 \
		--learning_rate 1e-5 \
		--warmup_steps 1000 \
		--num_train_epochs 10 \
		--gradient_accumulation_steps 1 \
		--evaluate_during_training \
		--checkpoint_path ./model/codesearchnet-checkpoint \
		--output_dir ${model} \
		--encoder_name_or_path microsoft/codebert-base \
        2>&1 | tee ./qa-train-codebert-coclr-${qra}.log
```

#### Step 4: evaluate on CodeXGLUE - code search (WebQueryTest)

You can submit the `--test_predictions_output` to CodeXGLUE challenge for the results on the test set.

```
qra=switch
model=./model/qa_codebert_${qra}
CUDA_VISIBLE_DEVICES="0" python ./code_qa/run_siamese_test.py \
		--model_type roberta  \
		--augment \
		--do_test \
		--data_dir ./data/qa \
		--test_data_file test_webquery.json \
		--max_seq_length 200 \
		--per_gpu_eval_batch_size 2 \
		--output_dir ${model}/checkpoint-best-aver/ \
		--encoder_name_or_path microsoft/codebert-base \
		--pred_model_dir ${model}/checkpoint-best-aver/ \
		--test_predictions_output ${model}/webquery_predictions.txt \
		2>&1| tee ./qa-test-codebert-coclr-${qra}.log
```

## Code Search

### Vanilla Model

#### Step 1: download the checkpoint trained on CodeSearchNet

Please to the first point in [Model Checkpoint section](#model-checkpoint)

#### Step 2: training and evaluating

To train a search model without CoCLR, you can use the following command:

```
model=./model/search_codebert
CUDA_VISIBLE_DEVICES="0" python ./code_search/run_siamese_test.py \
		--model_type roberta \
		--do_train \
		--do_eval \
		--eval_all_checkpoints \
        --data_dir ./data/search/ \
		--train_data_file cosqa-retrieval-train-19604.json \
		--eval_data_file cosqa-retrieval-dev-500.json \
		--retrieval_code_base code_idx_map.txt \
		--code_type code \
		--max_seq_length 200 \
		--per_gpu_train_batch_size 32 \
		--per_gpu_retrieval_batch_size 67 \
		--learning_rate 1e-6 \
		--num_train_epochs 10 \
		--gradient_accumulation_steps 1 \
		--evaluate_during_training \
		--checkpoint_path ./model/codesearchnet-checkpoint \
        --output_dir ${model} \
        --encoder_name_or_path microsoft/codebert-base \
        2>&1 | tee ./search-train-codebert.log

```

You can evaluate the model on the test set with the following command:

```
CUDA_VISIBLE_DEVICES="0" python ./code_search/run_siamese_test.py \
		--model_type roberta \
		--do_retrieval \
		--data_dir ./data/search/ \
		--test_data_file cosqa-retrieval-test-500.json \
		--retrieval_code_base code_idx_map.txt \
		--code_type code \
		--max_seq_length 200 \
		--per_gpu_retrieval_batch_size 67 \
		--output_dir ${model}/checkpoint-best-mrr/ \
		--encoder_name_or_path microsoft/codebert-base \
		--pred_model_dir ${model}/checkpoint-best-mrr \
		--retrieval_predictions_output ${model}/retrieval_outputs.txt \
		2>&1 | tee ./test-retrieval.log
```

### CoCLR on Code Search

To apply CoCLR on the task of code search, you can run the commands with the following steps.

#### Step 1: download the checkpoint trained on CodeSearchNet

Please to the first point in [Model Checkpoint section](#model-checkpoint)

#### Step 2: create query-rewritten data

```
cd data
python augment_qra.py --task retrieval --qra_mode delete
python augment_qra.py --task retrieval --qra_mode copy
python augment_qra.py --task retrieval --qra_mode switch
cd ../
```

#### Step 3: training and evaluating

```
qra=switch
model=./model/search_codebert_${qra}
CUDA_VISIBLE_DEVICES="0" python ./code_search/run_siamese_test.py \
		--model_type roberta \
		--augment \
		--do_train \
		--do_eval \
		--eval_all_checkpoints \
        --data_dir ./data/search/ \
		--train_data_file cosqa-retrieval-train-19604-qra-${qra}-28624.json \
		--eval_data_file cosqa-retrieval-dev-500.json \
		--retrieval_code_base code_idx_map.txt \
		--code_type code \
		--max_seq_length 200 \
		--per_gpu_train_batch_size 32 \
		--per_gpu_retrieval_batch_size 67 \
		--learning_rate 1e-6 \
		--num_train_epochs 10 \
		--gradient_accumulation_steps 1 \
		--evaluate_during_training \
		--checkpoint_path ./model/codesearchnet-checkpoint \
        --output_dir ${model} \
        --encoder_name_or_path microsoft/codebert-base \
        2>&1 | tee ./search-train-codebert-coclr-${qra}.log

CUDA_VISIBLE_DEVICES="0" python ./code_search/run_siamese_test.py \
		--model_type roberta \
		--do_retrieval \
		--data_dir ./data/search/ \
		--test_data_file cosqa-retrieval-test-500.json \
		--retrieval_code_base code_idx_map.txt \
		--code_type code \
		--max_seq_length 200 \
		--per_gpu_retrieval_batch_size 67 \
		--output_dir ${model}/checkpoint-best-mrr/ \
		--encoder_name_or_path microsoft/codebert-base \
		--pred_model_dir ${model}/checkpoint-best-mrr \
		--retrieval_predictions_output ${model}/retrieval_outputs.txt \
		2>&1 | tee ./search-test-codebert-coclr-${qra}.log
```

### Ablation with Model Component

To see the effects of different component of code in code search, we provide to run the ablation study. You can first create the test set of codes that some parts are removed, and then evaluate on these dataset with the following commands. You can select `--mode` with `header_only`, `doc_only`, `body_only`, `no_header`, `no_doc`, `no_body`.

```
cd data/search
python split_code_for_retrieval.py
mode=no_doc
CUDA_VISIBLE_DEVICES="0" python ./code_search/run_siamese_test.py \
		--model_type roberta \
		--do_retrieval \
		--data_dir ./data/search/ablation_test_code_component/${mode} \
		--test_data_file cosqa-retrieval-test-500.json \
		--retrieval_code_base code_idx_map.txt \
		--code_type code \
		--max_seq_length 200 \
		--per_gpu_retrieval_batch_size 67 \
		--output_dir ${model}/checkpoint-best-mrr/ \
		--encoder_name_or_path microsoft/codebert-base \
		--pred_model_dir ${model}/checkpoint-best-mrr \
		--retrieval_predictions_output ${model}/retrieval_outputs.txt \
		2>&1 | tee ./search-test-ablation-codebert-coclr-${qra}-${mode}.log
```

## Model Checkpoint

1. The checkpoint trained on CodeSearchNet can be downloaded through [this link](https://drive.google.com/drive/folders/1rM5A6dPf05Q5mP9kWjfsdRIpsqfI4IBi?usp=sharing). You can first download the checkpoint. Then move it to `./model/` and rename the dirname to `codesearchnet-checkpoint`. You can also use the data in [CodeXGLUE code search (WebQueryTest)](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-WebQuery/data) to train the models by your self.

2. The checkpoint with best code question answering results can be downloaded through [this link](https://drive.google.com/drive/folders/1VjZOEI_N25R_30ZL2hYNaY-43FfpQ_MD?usp=sharing) and move to `./model/`.

3. The checkpoint with best code search results can be downloaded through [this link](https://drive.google.com/drive/folders/1rmyqG68nmnjSFg4t8ywaSwJBCluKE4l2?usp=sharing) and move to `./model/`.

## Reference

If you find this project useful, please cite it using the following format:

```
@inproceedings{Huang2020CoSQA,
  title={CoSQA: 20, 000+ Web Queries for Code Search and Question Answering},
  author={Junjie Huang and Duyu Tang and Linjun Shou and Ming Gong and Ke Xu and Daxin Jiang and Ming Zhou and Nan Duan},
  booktitle={ACL/IJCNLP},
  year={2020}
}

@article{Lu2021CodeXGLUE,
  title={CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation},
  author={Shuai Lu and Daya Guo and Shuo Ren and Junjie Huang and Alexey Svyatkovskiy and Ambrosio Blanco and Colin Clement and Dawn Drain and Daxin Jiang and Duyu Tang and Ge Li and Lidong Zhou and Linjun Shou and Long Zhou and Michele Tufano and Ming Gong and Ming Zhou and Nan Duan and Neel Sundaresan and Shao Kun Deng and Shengyu Fu and Shujie Liu},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.04664}
}

```

