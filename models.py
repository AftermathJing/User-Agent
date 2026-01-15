import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import torch.distributed as dist


class UniversalPerceiverBlock(nn.Module):
    """
    单层 Perceiver Block: Cross Attention -> Self Attention -> FFN
    增加了 Dropout 以支持对比学习 (SimCSE)
    """

    def __init__(self, latent_dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads

        # 1. Cross Attention
        self.cross_attn_norm_q = nn.LayerNorm(latent_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(latent_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=heads, batch_first=True,
                                                dropout=dropout)

        # 2. Self Attention
        self.self_attn_norm = nn.LayerNorm(latent_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=heads, batch_first=True, dropout=dropout)

        # 3. FFN
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),  # FFN Dropout
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout)  # Output Dropout
        )
        self.dropout = nn.Dropout(dropout)  # Residual Dropout

    def forward(self, latents, context, context_mask=None):
        # --- Cross Attention ---
        q = self.cross_attn_norm_q(latents)
        k = v = self.cross_attn_norm_kv(context)
        key_padding_mask = (context_mask == 0) if context_mask is not None else None

        out_cross, _ = self.cross_attn(query=q, key=k, value=v, key_padding_mask=key_padding_mask)
        latents = latents + self.dropout(out_cross)

        # --- Self Attention ---
        q_sa = self.self_attn_norm(latents)
        out_sa, _ = self.self_attn(query=q_sa, key=q_sa, value=q_sa)
        latents = latents + self.dropout(out_sa)

        # --- FFN ---
        out_ffn = self.ffn(self.ffn_norm(latents))
        latents = latents + out_ffn  # Dropout is inside FFN seq

        return latents


class UniversalPersonaEncoder(nn.Module):
    def __init__(self, input_dim, universal_dim=1024, num_latents=32, num_layers=4, dropout=0.1, heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, universal_dim)
        self.universal_dim = universal_dim
        self.num_latents = num_latents

        self.latents = nn.Parameter(torch.randn(1, num_latents, universal_dim) * 0.02)
        self.pos_embed = nn.Embedding(4096, universal_dim)

        self.blocks = nn.ModuleList([
            UniversalPerceiverBlock(universal_dim, heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, history_embeds, attention_mask=None):
        B, L, D = history_embeds.shape

        # 1. 先投影：将维度统一到 universal_dim
        history_embeds = self.input_proj(history_embeds)

        # 2. 再加位置编码
        positions = torch.arange(L, device=history_embeds.device).unsqueeze(0).expand(B, -1)

        history_embeds = history_embeds + self.pos_embed(positions)

        # 3. 后续处理
        latents = self.latents.repeat(B, 1, 1)

        for block in self.blocks:
            latents = block(latents, history_embeds, attention_mask)

        return latents


class PersonaAdapter(nn.Module):
    def __init__(self, target_dim, universal_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(universal_dim),
            nn.Linear(universal_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
        nn.init.normal_(self.net[-2].weight, std=1e-4)  # 修改倒数第二层(Linear)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, universal_latents):
        return self.net(universal_latents)


class DifferentiableAllGather(torch.autograd.Function):
    """
    支持梯度的 AllGather，用于对比学习扩大 Batch Size
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()

        # 1. 准备接收列表
        # 注意：这里假设所有 GPU 上的 tensor 维度是一样的
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]

        # 2. 执行 all_gather
        dist.all_gather(gathered_tensors, tensor)

        # 3. 拼接
        return torch.cat(gathered_tensors, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，只取回属于当前进程的那一部分梯度
        # grad_output shape: [Batch * World, Dim]
        # 我们只需要切分出当前 rank 对应的那一段

        # 假设 batch size 是一致的
        per_card_batch = grad_output.shape[0] // ctx.world_size
        start = ctx.rank * per_card_batch
        end = start + per_card_batch

        return grad_output[start:end]


def all_gather_with_grad(tensor):
    """
    如果使用了 DDP，则进行 all_gather，否则直接返回本身
    """
    if dist.is_available() and dist.is_initialized():
        return DifferentiableAllGather.apply(tensor)
    return tensor


class PersonaAgent(nn.Module):
    def __init__(self, config):  # 传入整个 config 对象
        super().__init__()
        self.config = config
        self.cl_weight = config.cl_weight
        self.cl_temp = config.cl_temp

        print(f"Loading Base LLM: {config.base_model}...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"  # 或者由 DDP 控制 device
        )
        # 冻结 LLM
        for param in self.llm.parameters():
            param.requires_grad = False

        # 动态获取 LLM 维度
        llm_dim = self.llm.config.hidden_size

        # 使用 Config 初始化 Encoder
        self.encoder = UniversalPersonaEncoder(
            input_dim=llm_dim,  # 来自 LLM
            universal_dim=config.universal_dim,
            num_latents=config.num_latents,  # 新增
            num_layers=config.encoder_layers,  # 新增
            dropout=config.encoder_dropout,  # 新增
            heads=config.encoder_heads  # 需在 Encoder 内部支持传入 heads
        )

        # 使用 Config 初始化 Adapter
        self.adapter = PersonaAdapter(
            target_dim=llm_dim,
            universal_dim=config.universal_dim
        )

    def compute_contrastive_loss(self, view1, view2):
        """
        计算 InfoNCE Loss (Symmetric / CLIP style)
        支持多卡 Global Contrastive Loss
        """
        # 1. Pooling
        v1 = view1.mean(dim=1)
        v2 = view2.mean(dim=1)

        # 2. Normalize
        v1 = F.normalize(v1, dim=1)
        v2 = F.normalize(v2, dim=1)

        # --- 新增：全局收集 ---
        # 获取所有卡上的 embedding，拼接成大 Batch [Global_Batch, Dim]
        # 例如 4卡 * 4Batch = 16
        all_v1 = all_gather_with_grad(v1)
        all_v2 = all_gather_with_grad(v2)
        # --------------------

        # 3. 计算相似度矩阵
        # 本地样本 v1 (4个) 与 全局样本 all_v2 (16个) 计算相似度
        # sim_matrix shape: [Local_Batch, Global_Batch] -> [4, 16]
        sim_matrix = torch.matmul(v1, all_v2.T) / self.cl_temp

        # 对称方向: 本地 v2 与 全局 all_v1
        sim_matrix_t = torch.matmul(v2, all_v1.T) / self.cl_temp

        # 4. 生成标签
        # 正样本的位置在哪里？
        # 假设当前是 Rank 1 (第二张卡)，Local Batch=4
        # 全局列表是 [Rank0_Data, Rank1_Data, Rank2_Data, Rank3_Data]
        # 那么 Rank 1 的样本对应的正样本索引是 4, 5, 6, 7

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            local_batch_size = v1.size(0)
            # 偏移量：当前卡之前的卡有多少数据
            offset = rank * local_batch_size
            labels = torch.arange(local_batch_size, device=v1.device) + offset
        else:
            labels = torch.arange(v1.size(0), device=v1.device)

        # 5. 计算 Loss
        loss_i2t = F.cross_entropy(sim_matrix, labels)
        loss_t2i = F.cross_entropy(sim_matrix_t, labels)

        return (loss_i2t + loss_t2i) / 2

    @torch.no_grad()
    def generate_response(self, history_input_ids, history_attention_mask, instruction_input_ids, max_new_tokens=128):
        """
        用于在训练中进行采样测试。
        """
        # 1. 生成 Persona Soft Prompts
        history_embeds = self.llm.get_input_embeddings()(history_input_ids)

        # 强制转 Float32 (为了数值稳定，防止 NaN)
        history_embeds = history_embeds.to(torch.float32)

        # 这里的 mask 传递逻辑要和你 forward 里保持一致
        encoder_out = self.encoder(history_embeds, history_attention_mask)
        soft_prompts = self.adapter(encoder_out)

        # 2. 获取当前指令的 Embeddings
        instruction_embeds = self.llm.get_input_embeddings()(instruction_input_ids)

        # 强制转换 inputs_embeds 的类型，使其与 LLM 权重类型 (bfloat16) 一致
        soft_prompts = soft_prompts.to(self.llm.dtype)

        # 3. 拼接 Embeddings
        inputs_embeds = torch.cat([soft_prompts, instruction_embeds], dim=1)

        # 4. 构造 Attention Mask
        B, N_prompts, _ = soft_prompts.shape
        B, N_instr, _ = instruction_embeds.shape

        generation_mask = torch.ones((B, N_prompts + N_instr), device=inputs_embeds.device, dtype=torch.long)

        # 5. 生成
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=generation_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.llm.config.pad_token_id,
            eos_token_id=self.llm.config.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        return outputs

    def forward(self, history_input_ids, history_attention_mask, llm_input_ids, llm_labels, llm_attention_mask):
        # 1. 提取 Embedding
        with torch.no_grad():
            history_embeds = self.llm.get_input_embeddings()(history_input_ids)

        # 强制转 Float32 (为了数值稳定，防止 NaN)
        history_embeds = history_embeds.to(torch.float32)

        # 2. View 1 (用于生成的路径)
        latents_1 = self.encoder(history_embeds, history_attention_mask)
        soft_prompts = self.adapter(latents_1)

        # 3. 计算对比损失 (仅在训练模式且 Batch > 1 时)
        contrastive_loss = 0.0
        if self.training and history_input_ids.size(0) > 1:
            # View 2 (仅用于计算 Loss，利用 Encoder 内部 Dropout 生成不同视角)
            latents_2 = self.encoder(history_embeds, history_attention_mask)
            contrastive_loss = self.compute_contrastive_loss(latents_1, latents_2)

        # 4. 生成路径拼接
        with torch.no_grad():
            inputs_embeds = self.llm.get_input_embeddings()(llm_input_ids)

        soft_prompts = soft_prompts.to(self.llm.dtype)

        combined_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)

        # Mask & Labels 拼接
        B, N, _ = soft_prompts.shape
        prompt_mask = torch.ones((B, N), device=soft_prompts.device, dtype=llm_attention_mask.dtype)
        combined_mask = torch.cat([prompt_mask, llm_attention_mask], dim=1)

        prompt_labels = torch.full((B, N), -100, device=llm_labels.device, dtype=llm_labels.dtype)
        combined_labels = torch.cat([prompt_labels, llm_labels], dim=1)

        # 5. CLM Loss (重构损失)
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels
        )
        rec_loss = outputs.loss

        # 6. 总损失
        total_loss = rec_loss + (self.cl_weight * contrastive_loss)

        return total_loss, rec_loss, contrastive_loss