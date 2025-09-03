# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        """
        初始化模型配置参数。

        Args:
            dropout (float): Dropout概率，默认为0.0。
            bos_token_id (int): 序列开始标记的token ID，默认为1。
            eos_token_id (int): 序列结束标记的token ID，默认为2。
            hidden_act (str): 隐藏层激活函数类型，默认为'silu'。
            hidden_size (int): 隐藏层维度大小，默认为512。
            intermediate_size (int): 前馈网络中间层维度，若为None则使用默认计算方式。
            max_position_embeddings (int): 最大位置编码长度，默认为32768。
            num_attention_heads (int): 注意力头的数量，默认为8。
            num_hidden_layers (int): Transformer隐藏层数量，默认为8。
            num_key_value_heads (int): 键值注意力头数量，默认为2。
            vocab_size (int): 词汇表大小，默认为6400。
            rms_norm_eps (float): RMSNorm中的epsilon值，用于数值稳定性，默认为1e-05。
            rope_theta (int): RoPE位置编码的theta参数，默认为1000000.0。
            flash_attn (bool): 是否启用Flash Attention优化，默认为True。

            # MOE相关参数（仅在use_moe=True时有效）
            use_moe (bool): 是否启用MoE结构，默认为False。
            num_experts_per_tok (int): 每个token选择的专家数量，默认为2。
            n_routed_experts (int): 总的路由专家数量，默认为4。
            n_shared_experts (int): 共享专家数量，默认为1。
            scoring_func (str): 专家评分函数，默认为'softmax'。
            aux_loss_alpha (float): 辅助损失的权重系数，默认为0.1。
            seq_aux (bool): 是否在序列级别上计算辅助损失，默认为True。
            norm_topk_prob (bool): 是否对top-k专家的概率进行归一化，默认为True。

        Returns:
            无返回值，初始化实例属性。
        """
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """构造函数，接收输入维度和数值稳定参数

        Args:
            dim (int): 输入特征的维度大小
            eps (float, optional): 用于数值稳定的小常数，防止除零错误. 默认值为1e-5

        Returns:
            None: 该函数不返回任何值，仅用于初始化对象属性
        """
        super().__init__()
        # 存储数值稳定性参数（防止除零错误），默认1e-5
        self.eps = eps
        # 创建可学习的缩放权重参数，初始值为全1向量，维度与输入一致
        self.weight = nn.Parameter(torch.ones(dim))

    # 内部归一化计算方法
    def _norm(self, x):
        # 计算RMS归一化：x / sqrt(mean(x²) + eps)
        # x.pow(2)：对输入张量进行平方运算
        # .mean(-1, keepdim=True)：计算最后一维的均值，保持维度不变（输出shape与输入一致）
        # torch.rsqrt：计算平方根的倒数（等价于1/sqrt(x)）
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # 前向传播方法，接收输入张量x
    def forward(self, x):
        # 1. 将输入转换为float类型以提高数值精度
        # 2. 应用_norm方法进行RMS归一化
        # 3. 乘以可学习权重self.weight
        # 4. 转换回输入x的原始数据类型（保持类型一致性）
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """
    预计算用于旋转位置编码的复数频率值

    该函数生成余弦和正弦频率值，用于实现旋转位置编码(RoPE)。
    通过预计算这些值，可以在后续的注意力计算中快速应用位置编码。

    Args:
        dim (int): 嵌入维度大小，必须为偶数
        end (int): 序列最大长度，默认为32768
        theta (float): 频率基数参数，默认为1e6

    Returns:
        tuple: 包含两个张量的元组
            - freqs_cos (Tensor): 余弦频率值张量，形状为[end, dim]
            - freqs_sin (Tensor): 正弦频率值张量，形状为[end, dim]
    """
    # 计算每个维度对应的频率值
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 创建位置索引序列
    t = torch.arange(end, device=freqs.device)

    # 计算每个位置和每个维度的频率乘积
    freqs = torch.outer(t, freqs).float()

    # 扩展余弦和正弦频率值以匹配嵌入维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码到查询和键向量上

    该函数实现旋转位置编码（Rotary Positional Embedding），通过将位置信息编码到注意力机制中的
    查询（q）和键（k）向量来增强模型对序列位置的感知能力。

    Args:
        q (torch.Tensor): 查询向量，形状通常为 [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): 键向量，形状通常为 [batch_size, seq_len, num_heads, head_dim]
        cos (torch.Tensor): 余弦位置编码值
        sin (torch.Tensor): 正弦位置编码值
        position_ids (torch.Tensor, optional): 位置索引，用于指定每个位置对应的编码
        unsqueeze_dim (int): 在应用编码时进行维度扩展的维度位置，默认为1

    Returns:
        tuple: 包含应用了旋转位置编码的查询和键向量
            - q_embed (torch.Tensor): 应用位置编码后的查询向量
            - k_embed (torch.Tensor): 应用位置编码后的键向量
    """
    def rotate_half(x):
        """
        旋转位置编码的核心操作，用于实现向量的旋转: 将输入张量在最后一个维度上分为两半，并交换它们的位置，前半部分取负号
        """
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 对查询和键向量分别应用旋转位置编码
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed



def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复张量中的键值对维度，用于扩展注意力机制中的键值头数量。

    该函数通过在指定维度上重复张量元素来实现键值头的扩展，等价于
    torch.repeat_interleave(x, dim=2, repeats=n_rep) 的功能。

    Args:
        x (torch.Tensor): 输入张量，形状为 (bs, slen, num_key_value_heads, head_dim)
                         其中 bs 为批次大小，slen 为序列长度，
                         num_key_value_heads 为键值头数量，head_dim 为头维度
        n_rep (int): 重复倍数，指定每个键值头需要重复的次数

    Returns:
        torch.Tensor: 扩展后的张量，形状为 (bs, slen, num_key_value_heads * n_rep, head_dim)
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 通过添加新维度、扩展和重塑操作实现张量重复
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 确定键值对（KV）的头数：若未指定则与注意力头数相同，否则使用指定值
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # 本地注意力头数（查询头数）
        self.n_local_heads = args.num_attention_heads
        # 本地键值头数
        self.n_local_kv_heads = self.num_key_value_heads
        # 每个键值头需要重复的次数（用于匹配查询头数）
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个注意力头的维度（隐藏层大小 ÷ 注意力头数）
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 线性投影层：将输入映射到查询（Q）、键（K）、值（V）
        # Q投影：输出维度=查询头数×头维度
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # K投影：输出维度=键值头数×头维度
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # V投影：输出维度=键值头数×头维度
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出投影：将注意力结果映射回隐藏层大小
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # 注意力分数的dropout（正则化）
        self.attn_dropout = nn.Dropout(args.dropout)
        # 注意力输出的dropout（正则化）
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 判断是否使用Flash Attention加速：需PyTorch内置scaled_dot_product_attention且开启配置
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        # 解析输入的batch_size、序列长度
        bsz, seq_len, _ = x.shape
        # 对输入x进行线性投影，得到Q、K、V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 重塑Q、K、V的形状，分离出注意力头维度
        # Q形状：(bsz, seq_len, n_local_heads, head_dim)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # K形状：(bsz, seq_len, n_local_kv_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # V形状：(bsz, seq_len, n_local_kv_heads, head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码（RoPE）：将位置信息注入Q和K
        # TODO: 为什么是对Q、K添加位置编码而不是输入X？
        cos, sin = position_embeddings
        # 仅使用与当前序列长度匹配的位置编码
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现（推理时复用历史KV，避免重复计算）
        if past_key_value is not None:
            # 若存在历史KV，则拼接当前KV与历史KV
            # 拼接K：(bsz, past_seq_len + curr_seq_len, ...)
            xk = torch.cat([past_key_value[0], xk], dim=1)
            # 拼接V：(bsz, past_seq_len + curr_seq_len, ...)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # 若需缓存，则保存当前KV供后续使用
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序，为注意力计算做准备（将头维度提前）
        xq, xk, xv = (
            # Q形状：(bsz, n_local_heads, seq_len, head_dim)
            xq.transpose(1, 2),
            # K重复n_rep次以匹配查询头数，形状：(bsz, n_local_heads, seq_len, head_dim)
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            # V重复n_rep次以匹配查询头数，形状：(bsz, n_local_heads, seq_len, head_dim)
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            # 若使用Flash Attention且序列长度不为1（避免特殊情况）
            # 训练时用配置的dropout率，推理时为0
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None

            if attention_mask is not None:
                # 处理注意力掩码
                # 重塑掩码形状以匹配Flash Attention要求：(bsz, n_local_heads, seq_len, total_seq_len)
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                # 转为布尔型掩码（True表示需要屏蔽）
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            # 调用PyTorch内置的Flash Attention加速实现
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            # 不使用Flash Attention时，手动计算注意力
            # 计算注意力分数：Q与K的点积，除以头维度的平方根（缩放）
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 手动添加因果掩码，对角线以上（未来位置）被屏蔽，上三角部分设为负无穷(-inf)，避免关注未来位置
            # scores+mask, 扩展维度以匹配scores：(1, 1, seq_len, seq_len)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # 应用注意力掩码（若存在，如屏蔽padding token）
            if attention_mask is not None:
                # 扩展掩码维度：(bsz, 1, 1, total_seq_len) → (bsz, 1, 1, total_seq_len)
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # 掩码位置设为-1e9（softmax后接近0）
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                # 加到注意力分数上
                scores = scores + extended_attention_mask

            # 计算注意力权重（softmax）并应用dropout
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 注意力权重与V相乘，得到注意力输出，形状：(bsz, n_local_heads, seq_len, head_dim)
            output = scores @ xv

        # 调整输出维度并映射回隐藏层大小，形状：(bsz, seq_len, n_local_heads×head_dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)

        # 输出投影+dropout，最终形状：(bsz, seq_len, hidden_size)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        """
        初始化MiniMind模型的配置参数和网络层。

        Args:
            config (MiniMindConfig): 包含模型配置信息的对象，包括隐藏层大小、中间层大小、
                                   dropout率和激活函数类型等参数
        """
        super().__init__()
        if config.intermediate_size is None:
            # 未指定中间层大小，则根据隐藏层大小自动计算并调整为64的倍数
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # 定义门控投影层，用于将隐藏状态映射到中间表示
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 定义下投影层，用于将中间表示映射回隐藏状态
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 定义上投影层，用于将隐藏状态映射到中间表示
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.dropout)
        # 获取并设置激活函数
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
