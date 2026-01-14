import json
import torch
from torch.utils.data import Dataset, Sampler
import random
from collections import defaultdict


class SocialPersonaDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_history_len=2048, max_target_len=512, max_history_items=50):
        """
        Args:
            data_path: JSONL文件路径
            tokenizer: HuggingFace Tokenizer
            max_history_len: 历史记录总Token限制
            max_target_len: 目标回复Token限制
            max_history_items: 最多选取多少条历史记录
        """
        self.tokenizer = tokenizer
        self.max_history_len = max_history_len
        self.max_target_len = max_target_len
        self.max_history_items = max_history_items
        self.data = []

        # 加载数据
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.data.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: {data_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 获取 UID，如果没有则默认为 unknown (仅针对 mock data 或异常数据)
        uid = item.get('uid', f'unknown_{idx}')

        history_list = item.get('history', [])
        current_context = item.get('context', "")
        target_response = item.get('target', "")

        # 1. 处理历史记录 (History)
        selected_history = history_list[-self.max_history_items:]
        history_text = "\n".join(selected_history)

        history_tokens = self.tokenizer(
            history_text,
            max_length=self.max_history_len,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # 2. 处理当前对话
        instruction_text = f"<|im_start|>user\n{current_context}<|im_end|>\n<|im_start|>assistant\n"
        full_text = instruction_text + target_response + "<|im_end|>"

        target_tokens = self.tokenizer(
            full_text,
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        )

        # 3. 构造 Labels
        input_ids = target_tokens.input_ids[0]
        labels = input_ids.clone()

        instruction_len = len(self.tokenizer(instruction_text, add_special_tokens=False).input_ids)
        if instruction_len < len(labels):
            labels[:instruction_len] = -100

        return {
            "uid": uid,  # 新增：返回 UID
            "history_input_ids": history_tokens.input_ids[0],
            "history_attention_mask": history_tokens.attention_mask[0],
            "llm_input_ids": input_ids,
            "llm_labels": labels,
            "llm_attention_mask": target_tokens.attention_mask[0]
        }


def collate_fn(batch):
    """
    自定义 batch 处理
    """
    from torch.nn.utils.rnn import pad_sequence

    # 新增：收集 UID 列表 (字符串列表，不进行 padding)
    uids = [item['uid'] for item in batch]

    history_ids = [item['history_input_ids'] for item in batch]
    history_masks = [item['history_attention_mask'] for item in batch]
    llm_ids = [item['llm_input_ids'] for item in batch]
    llm_labels = [item['llm_labels'] for item in batch]
    llm_masks = [item['llm_attention_mask'] for item in batch]

    pad_val = 0
    label_pad_val = -100

    return {
        "uids": uids,  # 传递给训练循环
        "history_input_ids": pad_sequence(history_ids, batch_first=True, padding_value=pad_val),
        "history_attention_mask": pad_sequence(history_masks, batch_first=True, padding_value=0),
        "llm_input_ids": pad_sequence(llm_ids, batch_first=True, padding_value=pad_val),
        "llm_labels": pad_sequence(llm_labels, batch_first=True, padding_value=label_pad_val),
        "llm_attention_mask": pad_sequence(llm_masks, batch_first=True, padding_value=0)
    }


class UniqueUserBatchSampler(Sampler):
    """
    自定义采样器：保证同一个 Batch 内没有重复的 UID。
    这对于对比学习（InfoNCE）至关重要，避免 False Negative。
    """

    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 预先按 UID 分组所有索引
        self.user_to_indices = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            uid = item.get('uid', f'unknown_{idx}')
            self.user_to_indices[uid].append(idx)

    def __iter__(self):
        # 1. 深度拷贝一份索引（因为我们要不断 pop）
        # 或者打乱每个用户的列表，用指针遍历
        user_queues = {
            uid: list(indices) for uid, indices in self.user_to_indices.items()
        }
        for uid in user_queues:
            random.shuffle(user_queues[uid])

        while True:
            batch = []
            available_users = [uid for uid, q in user_queues.items() if len(q) > 0]

            # 如果剩余用户不足以填满一个 batch (且 drop_last=True)，则结束
            if len(available_users) < self.batch_size and self.drop_last:
                break
            # 如果完全没数据了
            if not available_users:
                break

            # 随机选取 Batch Size 个用户
            # 注意：如果剩余用户少于 Batch Size 但 drop_last=False，这就变成了当前剩下所有用户
            k = min(len(available_users), self.batch_size)
            selected_users = random.sample(available_users, k)

            for uid in selected_users:
                idx = user_queues[uid].pop()  # 取出该用户的一个样本
                batch.append(idx)

            yield batch

    def __len__(self):
        # 估算 Batch 数量
        return len(self.dataset) // self.batch_size