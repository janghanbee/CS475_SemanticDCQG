experiment="hotpot_sub" #choose from: baseline hotpot_sub hotpot_comp
input_data_folder="subq_rel" #choose from: baseline subq_rel
# empty cache if needed

output_path="/dockerdata/siyao/ACS-QG/Datasets/output/QG/generated/$experiment/"
data_file_prefix="test"
st_idx=0
ed_idx=1000

CUDA_VISIBLE_DEVICES=1 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
    --model_type gpt2 \
    --model_name_or_path /dockerdata/siyao/ACS-QG/Datasets/output/QG/models/$experiment/ \
    --model_name  /dockerdata/siyao/ACS-QG/Datasets/output/QG/models/$experiment/checkpoint_mymodel_1.pth \
    --filename "/dockerdata/siyao/ACS-QG/Datasets/original/HotpotQA/$input_data_folder/${data_file_prefix}.json" \
    --filecache "/dockerdata/siyao/ACS-QG/Datasets/output/QG/cache/$experiment/${data_file_prefix}.${st_idx}_${ed_idx}.cache.pth" \
    --data_type $experiment \
    --output_file "$output_path${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.json" \
    --subq_dir "/dockerdata/siyao/ACS-QG/Datasets/output/QG/generated/hotpot_sub/${data_file_prefix}.${st_idx}_${ed_idx}.generated.gpt2.intermediate.json" 
