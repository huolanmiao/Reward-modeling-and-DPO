# MODEL_NAME_OR_PATH="../../models/Qwen-0.5B-Instruct" # model path

MODEL_NAME_OR_PATH="../output/rm/slice_end"
# TRAIN_DATASETS="../../datasets/PKU-SafeRLHF-single-dimension" # dataset path
TRAIN_DATASETS= "None"
TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
TRAIN_SPLIT="train" # split the dataset

EVAL_DATASETS="../../datasets/PKU-SafeRLHF-single-dimension" # dataset path
EVAL_TEMPLATE="PKUSafeRLHF" # dataset template
EVAL_SPLIT="test" # split the dataset

OUTPUT_DIR="../output/rm" # output dir

# For wandb online logging
export WANDB_API_KEY="700f0888a77741893374797d582bb8ecb3f4b353"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_split ${TRAIN_SPLIT} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_interval 100 \
     --epochs 3