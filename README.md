# etrag
The source code for the paper "End-to-End Trainable Retrieval-Augmented Generation for Relation Extraction".

## Base code

The base code is from the [SuRE](https://github.com/luka-group/SuRE) repository. We have made some modifications to the original code to make it compatible with the our model.

## Data preparation
Follow the instructions in the [SuRE](https://github.com/luka-group/SuRE) repository to prepare the data.
For the benchmark datasets, we provide the preprocessed data in the `data` directory.

## Requirements
<!-- Docker environment -->
We provide a Dockerfile to build the environment. To build the Docker image, run the following command:
```bash
docker build -t etrask .
```

## Quick start
To train the model and evaluate it, run the following command:
```bash
# TACRED 100% data
docker run --gpus all -v $(pwd):/workspace -it etrask python run_pretrained_aug_tag_eval.py --do_train --do_eval --do_test --output_dir model --model_name_or_path google/flan-t5-large --train_file data/tacred/v0_1.0/train.json --validation_file data/tacred/v0_1.0/dev.json --test_file data/tacred/v0_1.0/test.json --type_file data/tacred/types/type.json --type_constraint_file data/tacred/types/type_constraint.json --template_file data/templates/tacred/rel2temp.json --text_column text --summary_column target --max_source_length 272 --min_target_length 0 --max_target_length 64 --learning_rate 5e-4 --weight_decay 5e-6 --num_beams 4 --num_train_epochs 1000 --preprocessing_num_workers 8 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --num_warmup_steps 0 --seed 100 --eval_loss_only --eval_steps 100 --early_stopping --early_stopping_patience 3 --wandb_group tacred-1.0-t5-large-knn-warmup --run_name tacred-1.0-t5-large-knn-warmup-seed_100 --lora --lora_r 32 --lora_alpha 16 --lora_dropout 0.1 --embedding_retriever knn --embedding_retriever_batch_size 32 --knn_embedding_retriever_max_sampling 32 --embedding_retriever_sampling 5000 --embedding_retriever_update_step 100 --embedding_retriever_model roberta-base -n 10 --knn_embedding_retriever_differentiable --embedding_retriever_detect_span --knn_embedding_retriever_temperature 1 --knn_embedding_retriever_distance_reduction mean --embedding_retriever_insert --embedding_retriever_warmup 300 --early_stopping_warmup 3
# TACRED 10% data
docker run --gpus all -v $(pwd):/workspace -it etrask python run_pretrained_aug_tag_eval.py --do_train --do_eval --do_test --output_dir model --model_name_or_path google/flan-t5-large --train_file data/tacred/v0_0.1/train.json --validation_file data/tacred/v0_0.1/dev.json --test_file data/tacred/v0_0.1/test.json --type_file data/tacred/types/type.json --type_constraint_file data/tacred/types/type_constraint.json --template_file data/templates/tacred/rel2temp.json --text_column text --summary_column target --max_source_length 272 --min_target_length 0 --max_target_length 64 --learning_rate 5e-4 --weight_decay 5e-6 --num_beams 4 --num_train_epochs 1000 --preprocessing_num_workers 8 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --num_warmup_steps 0 --seed 100 --eval_loss_only --eval_steps 100 --early_stopping --early_stopping_patience 3 --wandb_group tacred-0.1-t5-large-knn-warmup --run_name tacred-0.1-t5-large-knn-warmup-seed_100 --lora --lora_r 32 --lora_alpha 16 --lora_dropout 0.1 --embedding_retriever knn --embedding_retriever_batch_size 32 --knn_embedding_retriever_max_sampling 32 --embedding_retriever_sampling 5000 --embedding_retriever_update_step 100 --embedding_retriever_model roberta-base -n 10 --knn_embedding_retriever_differentiable --embedding_retriever_detect_span --knn_embedding_retriever_temperature 1 --knn_embedding_retriever_distance_reduction mean --embedding_retriever_insert --embedding_retriever_warmup 300 --early_stopping_warmup 3
# TACRED 5% data
docker run --gpus all -v $(pwd):/workspace -it etrask python run_pretrained_aug_tag_eval.py --do_train --do_eval --do_test --output_dir model --model_name_or_path google/flan-t5-large --train_file data/tacred/v0_0.05/train.json --validation_file data/tacred/v0_0.05/dev.json --test_file data/tacred/v0_0.05/test.json --type_file data/tacred/types/type.json --type_constraint_file data/tacred/types/type_constraint.json --template_file data/templates/tacred/rel2temp.json --text_column text --summary_column target --max_source_length 272 --min_target_length 0 --max_target_length 64 --learning_rate 5e-4 --weight_decay 5e-6 --num_beams 4 --num_train_epochs 1000 --preprocessing_num_workers 8 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --num_warmup_steps 0 --seed 100 --eval_loss_only --eval_steps 100 --early_stopping --early_stopping_patience 3 --wandb_group tacred-0.05-t5-large-knn-warmup --run_name tacred-0.05-t5-large-knn-warmup-seed_100 --lora --lora_r 32 --lora_alpha 16 --lora_dropout 0.1 --embedding_retriever knn --embedding_retriever_batch_size 32 --knn_embedding_retriever_max_sampling 32 --embedding_retriever_sampling 5000 --embedding_retriever_update_step 100 --embedding_retriever_model roberta-base -n 10 --knn_embedding_retriever_differentiable --embedding_retriever_detect_span --knn_embedding_retriever_temperature 1 --knn_embedding_retriever_distance_reduction mean --embedding_retriever_insert --embedding_retriever_warmup 300 --early_stopping_warmup 3
# TACRED 1% data
docker run --gpus all -v $(pwd):/workspace -it etrask python run_pretrained_aug_tag_eval.py --do_train --do_eval --do_test --output_dir model --model_name_or_path google/flan-t5-large --train_file data/tacred/v0_0.01/train.json --validation_file data/tacred/v0_0.01/dev.json --test_file data/tacred/v0_0.01/test.json --type_file data/tacred/types/type.json --type_constraint_file data/tacred/types/type_constraint.json --template_file data/templates/tacred/rel2temp.json --text_column text --summary_column target --max_source_length 272 --min_target_length 0 --max_target_length 64 --learning_rate 5e-4 --weight_decay 5e-6 --num_beams 4 --num_train_epochs 1000 --preprocessing_num_workers 8 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --num_warmup_steps 0 --seed 100 --eval_loss_only --eval_steps 100 --early_stopping --early_stopping_patience 3 --wandb_group tacred-0.01-t5-large-knn-warmup --run_name tacred-0.01-t5-large-knn-warmup-seed_100 --lora --lora_r 32 --lora_alpha 16 --lora_dropout 0.1 --embedding_retriever knn --embedding_retriever_batch_size 32 --knn_embedding_retriever_max_sampling 32 --embedding_retriever_sampling 5000 --embedding_retriever_update_step 100 --embedding_retriever_model roberta-base -n 10 --knn_embedding_retriever_differentiable --embedding_retriever_detect_span --knn_embedding_retriever_temperature 1 --knn_embedding_retriever_distance_reduction mean --embedding_retriever_insert --embedding_retriever_warmup 300 --early_stopping_warmup 3
```
