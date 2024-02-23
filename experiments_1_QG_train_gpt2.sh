experiment=hotpot_comp #choose from: baseline hotpot_sub hotpot_comp
input_data_folder=subq_rel #choose from: baseline subq_rel
# output dir need to be empty before running
# empty cache if needed

CUDA_VISIBLE_DEVICES=0,1 python QG_gpt2_train.py \
    --eval_before_start \
    --n_epochs 2 \
    --model_name_or_path /dockerdata/siyao/transformer_models/gpt2 \
    --output_dir /dockerdata/siyao/ACS-QG/Datasets/output/QG/models/$experiment/ \
    --train_dataset_path /dockerdata/siyao/ACS-QG/Datasets/original/HotpotQA/$input_data_folder/dev.json \
    --dev_dataset_path /dockerdata/siyao/ACS-QG/Datasets/original/HotpotQA/$input_data_folder/test.json \
    --train_dataset_cache /dockerdata/siyao/ACS-QG/Datasets/output/QG/cache/$experiment/dev_hotpot_cache \
    --dev_dataset_cache /dockerdata/siyao/ACS-QG/Datasets/output/QG/cache/$experiment/test_hotpot_cache \
    --filetype $experiment

Interrogative words (e.g., “when”, “how”, and “why”) play a vital role in QG
Mismatch between generated question words and answer type is common for seq2seq NLG systems

