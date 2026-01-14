# 核心参数解释：
# CUDA_VISIBLE_DEVICES=0,1,2,3  -> 指定程序只能看到前4张卡
# --nproc_per_node=4            -> 启动4个进程（对应4张卡）
# --master_port=29500           -> 指定通信端口（防止端口冲突，可选）

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 train.py \
    --base_model "/home/wj/Qwen3-8B" \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --save_dir "./checkpoints_test"