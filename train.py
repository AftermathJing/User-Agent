import os
import argparse
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

# 引入你项目中的模块
from models import PersonaAgent
from dataset import SocialPersonaDataset, collate_fn, UniqueUserBatchSampler


# --- 1. 配置管理 ---
@dataclass
class TrainingConfig:
    # 基础模型配置
    base_model: str = "Qwen/Qwen3-8B"
    data_path: str = "./train.jsonl"
    output_dir: str = "./checkpoints"
    run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 训练超参
    epochs: int = 3
    batch_size: int = 4  # 单卡 Batch Size (全局 BS = batch_size * num_gpus)
    grad_accum_steps: int = 4  # 梯度累积
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0  # 梯度裁剪

    # 数据相关
    max_history_len: int = 2048
    num_workers: int = 4

    # 模型特定参数 (对比学习)
    cl_weight: float = 0.1
    cl_temp: float = 0.05
    universal_dim: int = 1024

    # 系统
    seed: int = 42
    log_interval: int = 10  # 多少步记录一次 TensorBoard
    save_interval_steps: int = 500  # 多少步保存一次 checkpoint (可选)


def parse_args():
    parser = argparse.ArgumentParser()
    # 简单起见，这里只列出部分核心参数，实际使用可以把上面的Config全部映射
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    return parser.parse_args()


# --- 2. 工具函数 ---
def setup_distributed():
    """初始化 DDP 环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Rank {rank}/{world_size} initialized (Local Rank: {local_rank})")
        return rank, world_size, local_rank
    else:
        print("Not using distributed mode.")
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- 3. 训练主逻辑 ---
def train():
    # A. 初始化
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_master = (rank == 0)

    # 合并 args 到 config (此处简化处理，实际可用专门的配置库)
    config = TrainingConfig(
        base_model=args.base_model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=args.save_dir
    )

    if is_master:
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        # TensorBoard Writer
        log_dir = os.path.join(config.output_dir, "logs", config.run_name)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to: {log_dir}")
        print(f"Global Batch Size: {config.batch_size * world_size * config.grad_accum_steps}")

    set_seed(config.seed + rank)  # 不同 Rank 使用不同随机种子 (虽然数据切分了，但 Dropout 等需要随机性)

    # B. 加载 Tokenizer
    if is_master:
        print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # C. 加载数据集 & DDP 分片处理
    # 核心逻辑：为了保留 UniqueUserBatchSampler 的功能，我们在 Dataset 层面进行切分
    # 而不是使用 DistributedSampler。这样每个 GPU 拿到不同的用户子集。
    dataset = SocialPersonaDataset(
        config.data_path,
        tokenizer,
        max_history_len=config.max_history_len
    )

    # --- DDP Data Sharding ---
    # 手动切分 dataset.data，让每个 rank 只看一部分数据
    total_data = len(dataset.data)
    indices = list(range(total_data))
    # 简单的按顺序切分 (0,1,2 -> rank0; 3,4,5 -> rank1 ...)
    # 也可以 shuffle 后切分，取决于是否需要跨 Epoch 随机 (这里保持简单，且 UniqueUserBatchSampler 内部会 shuffle)
    my_indices = indices[rank::world_size]
    dataset.data = [dataset.data[i] for i in my_indices]

    if is_master:
        print(f"Total data: {total_data}. Rank 0 data size: {len(dataset.data)}")

    # 实例化 Sampler (作用于切分后的局部数据集)
    unique_sampler = UniqueUserBatchSampler(dataset, batch_size=config.batch_size, drop_last=True)

    dataloader = DataLoader(
        dataset,
        batch_sampler=unique_sampler,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # D. 初始化模型
    if is_master:
        print("Initializing Model...")

    # 确保模型加载到对应 GPU
    agent = PersonaAgent(
        config.base_model,
        universal_dim=config.universal_dim,
        cl_weight=config.cl_weight,
        cl_temp=config.cl_temp
    )

    # 这里的 .to() 很重要，确保 module 在 wrap DDP 之前在正确的 device 上
    device = torch.device(f"cuda:{local_rank}")
    agent = agent.to(device)

    # 开启训练模式
    agent.train()

    # DDP Wrapping
    # find_unused_parameters=True 因为 base_model (LLM) 被冻结，参数不参与更新
    agent = DDP(agent, device_ids=[local_rank], find_unused_parameters=True)

    # E. 优化器与 Scheduler
    # 过滤掉不需要梯度的参数
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    if is_master:
        print(f"Trainable Parameters: {len(trainable_params)}")

    optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)

    # 计算总步数
    steps_per_epoch = len(dataloader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps
    )

    # F. 训练循环
    global_step = 0

    for epoch in range(config.epochs):
        # 注意：使用自定义 BatchSampler 时，不需要调用 dataloader.sampler.set_epoch(epoch)
        # 因为我们的 Sharding 是静态的，Shuffle 由 Sampler 内部控制

        if is_master:
            print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")
            progress_bar = tqdm(total=steps_per_epoch, desc=f"Training", dynamic_ncols=True)

        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            # 1. 数据搬运
            # collate_fn 返回的 uids 是 list，不需要 .to(device)
            # 其他 tensor 放入 GPU
            model_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k != "uids"
            }

            # 2. Forward (Mixed Precision)
            # 使用 autocast 自动处理 bfloat16/float32 混合
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, rec_loss, cl_loss = agent(
                    history_input_ids=model_inputs['history_input_ids'],
                    history_attention_mask=model_inputs['history_attention_mask'],
                    llm_input_ids=model_inputs['llm_input_ids'],
                    llm_labels=model_inputs['llm_labels'],
                    llm_attention_mask=model_inputs['llm_attention_mask']
                )

                # 3. Loss Scale
                loss = loss / config.grad_accum_steps

            # 4. Backward
            loss.backward()

            # 5. Optimization Step
            if (step + 1) % config.grad_accum_steps == 0:
                # 梯度裁剪 (对齐 DDP 全局梯度)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # 6. Logging (Only Master)
                if is_master:
                    # 获取标量值 (处理可能为 Tensor 或 float 的情况)
                    rec_val = rec_loss.item()
                    cl_val = cl_loss.item() if isinstance(cl_loss, torch.Tensor) else cl_loss
                    total_val = loss.item() * config.grad_accum_steps  # 还原回真实Loss

                    if global_step % config.log_interval == 0:
                        writer.add_scalar("Loss/Total", total_val, global_step)
                        writer.add_scalar("Loss/Reconstruction", rec_val, global_step)
                        writer.add_scalar("Loss/Contrastive", cl_val, global_step)
                        writer.add_scalar("Training/LR", scheduler.get_last_lr()[0], global_step)

                    progress_bar.set_postfix({
                        'Loss': f"{total_val:.3f}",
                        'Rec': f"{rec_val:.3f}",
                        'CL': f"{cl_val:.3f}",
                        'LR': f"{scheduler.get_last_lr()[0]:.1e}"
                    })
                    progress_bar.update(1)

        # Epoch 结束保存
        if is_master:
            progress_bar.close()
            print(f"Saving Checkpoint for Epoch {epoch + 1}...")
            # 注意：保存时要取 .module 来获取原始模型，去除 DDP 包装
            save_path = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch + 1}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save(agent.module.encoder.state_dict(), os.path.join(save_path, "encoder.bin"))
            torch.save(agent.module.adapter.state_dict(), os.path.join(save_path, "adapter.bin"))
            tokenizer.save_pretrained(save_path)  # 保存 tokenizer 以方便推理

    cleanup_distributed()
    if is_master:
        print("Training Finished!")
        writer.close()


if __name__ == "__main__":
    # 设置 PyTorch 显存分配策略，减少碎片
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    train()