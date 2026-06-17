import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset, LLMDataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()

def rotate_half(x: torch.Tensor):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotate_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    k_embed = (k*cos) + (rotate_half(k)*sin)
    
    # 训练或prefill
    if q.shape[1] == k.shape[1]:
        q_embed = (q*cos) + (rotate_half(q)*sin)
    # 推理，逐个token
    else:
        q_embed = (q*cos[:,-1:]) + (rotate_half(q)*sin[:,-1:])
    
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, q, k):
        cos = self.cos_cached[:k.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:k.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn
        self.layer_idx = layer_idx
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
    def forward(self, hidden_states, 
                attention_mask: torch.Tensor=None, 
                cache_params=None):
        b, s = hidden_states.shape[:2]
        q, k, v, z = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states), self.gate_proj(hidden_states)
        
        if cache_params is not None:
            if cache_params.key_cache[self.layer_idx] is None:
                cache_params.key_cache[self.layer_idx] = k
                cache_params.value_cache[self.layer_idx] = v
            else:
                k = torch.cat([cache_params.key_cache[self.layer_idx], k], dim=1)
                v = torch.cat([cache_params.value_cache[self.layer_idx], v], dim=1)
                cache_params.key_cache[self.layer_idx] = k
                cache_params.value_cache[self.layer_idx] = v
        
        q = q.view(b, q.shape[1], self.num_heads, self.head_dim)
        k = k.view(b, k.shape[1], self.num_key_value_heads, self.head_dim)
        v = v.view(b, v.shape[1], self.num_key_value_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q, k = self.rotary_emb(q, k)
        
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        
        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 
        
        if self.flash_attn and s > 1:
            attn_mask = None
            if attention_mask is not None:
                mask = torch.triu(torch.ones(b, s, s), diagonal=1).bool().to(attention_mask.device) # [b,s,s]
                attention_mask = attention_mask.bool().unsqueeze(-1) # [b,s,1]
                attn_mask = mask | ~attention_mask 
                attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf')).unsqueeze(1)
            
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, 
                                                   dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal) 
        
        else:
            mask = torch.triu(torch.ones(b, s, s), diagonal=1).bool()
            if attention_mask is not None:
                mask = mask.to(attention_mask.device)
                mask = mask | ~attention_mask.bool().unsqueeze(-1)
            
            mask = mask.float().masked_fill(mask, float('-inf')).unsqueeze(1)
            
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数

            scores = scores + mask  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出
        
        output = output.transpose(1, 2).contiguous().view(b, s, -1) # b, s, self.hidden_size
        z = z.sigmoid()
        output = output * z
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output
        


class DynamicCache:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_states = [None for _ in range(config.n_layers)]
        self.recurrent_states = [None for _ in range(config.n_layers)]
        self.key_cache = [None for _ in range(config.n_layers)]
        self.value_cache = [None for _ in range(config.n_layers)]


class GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads    # 16
        self.num_k_heads = config.linear_num_key_heads      # 8
        self.head_k_dim = config.linear_key_head_dim        # 16
        self.head_v_dim = config.linear_value_head_dim      # 16
        self.key_dim = self.head_k_dim * self.num_k_heads   # 128
        self.value_dim = self.head_v_dim * self.num_v_heads # 256
        
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)
        
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNorm(self.value_dim)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, 
                cache_params: DynamicCache = None, 
                attention_mask: torch.Tensor = None):
        bs, s, _ = hidden_states.shape # 预训练阶段 s == 文本长度，此处 s = 511 ，预训练阶段 cache_params 始终为 None，只需要一次前向计算，不需要保存缓存
        
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)
            
        # [bs, seq_len, 2*(key_dim + value_dim)]
        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        # [bs, seq_len, 2*num_v_heads]
        projected_states_ba = self.in_proj_ba(hidden_states)
        
        # query.shape = key.shape =  [bs, seq_len, key_dim] = [16, 511, 128]
        # value.shape = z.shape = [bs, seq_len, value_dim] = [16, 511, 256]
        query, key, value, z = torch.split(
            projected_states_qkvz, 
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim], 
            dim = -1)
        # b.shape = a.shape = [bs, seq_len, num_v_heads] = [16, 511, 16]
        b, a = torch.split(
            projected_states_ba, 
            [self.num_v_heads, self.num_v_heads], 
            dim=-1)
        # mixed_qkv.sahpe = [bs, seq_len, 2*key_dim + value_dim] = [16, 511, 512]
        mixed_qkv = torch.cat([query, key, value], dim=-1)
        mixed_qkv = mixed_qkv.transpose(1,2) # shape = [bs, 2*key_dim +value_dim, seq_len] = [16, 512, 511]
        # 推理阶段
        if cache_params is not None:
            # 推理， prefill 阶段
            if s>1:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :s])
            # 推理，generate 阶段，每次只传入一个 token ，所以 s = 1
            else:
                # 写法 1
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.cat([conv_state, mixed_qkv], dim=-1)
                state_len = conv_state.shape[-1]
                
                mixed_qkv = F.silu(self.conv1d(conv_state)[:, :, state_len-1:state_len])
                cache_params.conv_states[self.layer_idx] = conv_state
                
                # # 写法 2
                # conv_state = cache_params.conv_states[self.layer_idx]
                # mixed_qkv = torch_causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias)
        # 训练阶段
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :s])
        # mixed_qkv.shape = [bs, seq_len, 2*key_dim +value_dim] = [16, 511, 512]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        # query.shape = k.shape = [bs, seq_len, key_dim] = [15, 511, 128],  v.shape = [bs, seq_len, value_dim] = [16, 511, 256]
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        
        query = query.reshape(bs, s, -1, self.head_k_dim)   #[bs, seq_len, num_k_heads, key_dim]   = [16, 511, 8, 16]
        key = key.reshape(bs, s, -1, self.head_k_dim)       #[bs, seq_len, num_k_heads, key_dim]   = [16, 511, 8, 16]
        value = value.reshape(bs, s, -1, self.head_v_dim)   #[bs, seq_len, num_v_heads, value_dim] = [16, 511, 16, 16]
        # b.shape = a.shape = [bs, seq_len, num_v_heads]
        beta = b.sigmoid()
        
        # 处理阿法，获取遗忘门， 遗忘门的获取方式不唯一 shape = [bs, seq_len, num_v_heads] = [16, 511, 16]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        
        # 下面的操作类似于 GQA, 保持 head 数量一致，经过此步骤 q.shape = k.sahpe = v.shape =[bs, seq_len, num_v_heads, head_v_dim]
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        # q.shape = k.sahpe = v.shape = [16, 511, 16, 16]
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
        # 经过下面操作后，q.shape = k.sahpe = v.shape =[bs, num_v_heads, seq_len, head_v_dim]  beta.shape = g.shape = [bs, num_v_heads, seq_len]
        query = query.transpose(1, 2).contiguous().to(torch.float32)
        key = key.transpose(1, 2).contiguous().to(torch.float32)
        value = value.transpose(1, 2).contiguous().to(torch.float32)
        beta = beta.transpose(1, 2).contiguous().to(torch.float32)
        g = g.transpose(1, 2).contiguous().to(torch.float32)
        # num_heads = num_v_heads = 16, head_k_dim = head_value_dim = 16
        _, num_heads, _, head_k_dim = key.shape
        head_v_dim = value.shape[-1]
        core_attn_out = torch.zeros(bs, num_heads, s, head_v_dim).to(value) # [bs, num_v_heads, seq_len, head_value_dim] = [16, 16, 511, 16]
        
        recurrent_state = None
        if cache_params is not None:
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        last_recurrent_state = torch.zeros(bs, num_heads, head_k_dim, head_v_dim).to(value) if recurrent_state is None else recurrent_state.to(value)
        # q.shape = k.sahpe = v.shape =[bs, num_v_heads, seq_len, head_v_dim]  beta.shape = g.shape = [bs, num_v_heads, seq_len]
        # q_t.shape = k_t.shape = v_t.shape = [bs, num_v_heads, head_v_dim]  g_t.shape = [bs, num_v_heads, 1, 1] beta_t.shape = [bs, num_v_heads, 1]
        for i in range(s):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)
            
            last_recurrent_state = last_recurrent_state * g_t                   # [bs, num_v_heads==num_heads, head_K_dim, head_v_dim]
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)     # [bs, num_v_heads==num_heads, head_v_dim]
            delta = (v_t - kv_mem) * beta_t                                     # [bs, num_v_heads==num_heads, head_v_dim]
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
            
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(query.dtype) # [bs, seq_len, num_v_heads, head_value_dim]
        core_attn_out = core_attn_out.reshape(bs, s, -1)                           # [bs, seq_len, value_dim]
        
        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
        
        z = F.silu(z)
        core_attn_out = self.norm(core_attn_out)
        core_attn_out = core_attn_out * z
        output = self.out_proj(core_attn_out)
        return output
            

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def load_balancing_loss_func(
    gate_logits,
    num_experts,
    top_k):
    concatenated_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0) # 各个层的gate_logit进行合并[layers X batch_size X sequence_length, num_experts]
    routing_weights = F.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)
        
    def forward(self, x):
        # x dim: b, s, hidden_size
        logits = self.gate(x)  # gate: b, s, expert_num
        logits_topk, indices = logits.topk(self.topk, dim=-1) # 选择概率最大的两个专家，返回两个专家对每个token的概率 [bs, seq_len, topk]
        zeros = torch.full_like(logits, float("-inf")) # 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率并重新归一化概率最大的两个专家
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 将选择的两个专家的概率按指定索引填充 [bs, seq_len, expert_nums] 填充后，除去 topk 的专家，剩下位置logits 都是 -inf
        sparse_logits = F.softmax(sparse_logits, dim=-1) # 得到一个稀疏矩阵，选择的两个专家对每个token的概率和为1 [bs, seq_len, expert_nums]
        gate_logit = logits.view(-1, self.expert_num)    # [bs * seq_len, expert_nums]
        
        return sparse_logits, indices, gate_logit

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias) 
    
    def forward(self, x): # x.shape = [选了第i个专家的所有token, hidden_size] = [seq_true, dim]
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)
        # x.shape = [bs, seq_len, hidden_size] = [16, 511, 256]
    def forward(self, x):
        sparse_logits, indices, gate_logit = self.gating(x) # sparse_logits.shape = [bs, seq_len, expert_nums] 其中仅有 topk 个位置是正常值，其他都是 -inf  indices = [bs, seq_len, topk]  gate_logits.shape = [bs * seq_len, expert_nums]
        final_outputs = torch.zeros_like(x)                 # [bs, seq_len, hidden_size]
        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, hidden_size)
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num))
        
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1) # (batch_size * seq_len)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (seq_true, dim)
                export_output = expert(expert_input)  # (seq_true, dim)
                
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)
                
                weighted_output = export_output * gate_scores  # (seq_true, dim)
                
                final_outputs[expert_mask] += weighted_output
                
        
        return final_outputs, gate_logit

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.moe = MoE(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        # self.post_attention_layernorm_zxd = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx
        self.softmax_attn_index = config.softmax_attn_index
        
        if layer_idx in self.softmax_attn_index:
            self.self_attn = Attention(config, layer_idx)
        else:
            self.linear_attn = GatedDeltaNet(config, layer_idx)
            
    def forward(
        self,
        hidden_states,
        attention_mask: torch.Tensor = None,
        cache_params:DynamicCache = None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.layer_idx in self.softmax_attn_index:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cache_params=cache_params
            )
        else:
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
            )
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.layer_idx % 2 == 0:
            hidden_states = self.mlp(hidden_states)
            gate_logit = None
        else:
            hidden_states, gate_logit = self.moe(hidden_states)
        outputs = residual + hidden_states
        return outputs, gate_logit

# 编写自定义配置时需要记住的三个重要事项如下：
# 1、必须继承自 PretrainedConfig
# 2、PretrainedConfig 的 __init__ 方法必须接受任何 kwargs
# 3、这些 kwargs 需要传递给超类的 __init__ 方法。
class Config(PretrainedConfig):
    model_type = 'moe_model'
    def __init__(self,
        hidden_size = 256,
        num_attention_heads = 16,
        num_key_value_heads = 4,
        linear_num_value_heads = 16,
        linear_num_key_heads = 8,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4,
        flash_attn = True,
        attention_bias = False,
        max_seq_len = 512,
        intermediate_size = 1024,
        mlp_bias = False,
        vocab_size = 6400,
        n_layers = 8,
        softmax_attn_index = [3,7],
        dropout = 0.0,
        expert_num = 8,
        topk = 2,
        output_router_logits = True,
        aux_loss_coef = 0.01,
        **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads # softmax注意力q头数
        self.num_key_value_heads = num_key_value_heads # softmax注意力k,v头数
        
        self.linear_num_value_heads = linear_num_value_heads # 线性注意力v的头数
        self.linear_num_key_heads = linear_num_key_heads # 线性注意力k的头数
        self.linear_key_head_dim = linear_key_head_dim # 线性注意力k每个头的维度
        self.linear_value_head_dim = linear_value_head_dim # 线性注意力v每个头的维度
        
        self.linear_conv_kernel_dim = linear_conv_kernel_dim # 线性注意力卷积核大小
        
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.softmax_attn_index = softmax_attn_index
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)

class LLM(PreTrainedModel):
    config_class = Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers
        self.expert_num = self.config.expert_num
        self.topk = self.config.topk

        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout) 
        self.layers = torch.nn.ModuleList() 
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx)) 
        self.norm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) 
        self.token_embeddings.weight = self.output.weight
        self.apply(self._init_weights) 
        self.loss = None 
        self.aux_loss = None

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)) 
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
    
    def forward(self, input_ids, attention_mask=None, labels=None, cache_params = None):
        # attention_mask.shape [bs, max_seq_len-1] # 后续用 max_seq_len 代替，因为是先 拼到 512 ，然后需要配置 label ，所以都少了一个长度, 三者形状一致
        # labels.shape [bs, max_seq_len-1] 
        # input_ids.shape = [bs=16, max_seq_len-1=511] 
        all_router_logits = () if self.config.output_router_logits else None
        
        hidden_states = self.token_embeddings(input_ids)  # [bs, seq_len, hidden_size]
        hidden_states = self.dropout(hidden_states)  
        for idx, layer in enumerate(self.layers):
            hidden_states, gate_logit = layer(hidden_states, attention_mask = attention_mask, cache_params=cache_params)
            if gate_logit is not None:
                all_router_logits += (gate_logit, )  
        hidden_states = self.norm(hidden_states) 
        
        if labels is not None:
            logits = self.output(hidden_states)  
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0) 
        else:
            logits = self.output(hidden_states[:, [-1], :])  
            self.loss = None  
        
        if self.config.output_router_logits:
            self.aux_loss = load_balancing_loss_func(all_router_logits, self.expert_num, self.topk)
            
            if labels is not None:
                self.loss += self.config.aux_loss_coef * self.aux_loss.to(self.loss.device)
        
        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1, use_cache=True):
        cache_params = None
        if use_cache:
            cache_params = DynamicCache(self.config)
        
        input_ids = inputs['input_ids']
        attention_mask = None
        labels = None
        s = input_ids.shape[1]
        
        idx_next = None
        while input_ids.shape[1] < max_new_tokens - 1:  
            if idx_next is None:
                inference_res = self(input_ids, attention_mask=attention_mask, labels=labels, cache_params=cache_params)  
            else:
                inference_res = self(idx_next, attention_mask=attention_mask, labels=labels, cache_params=cache_params)
                
            logits = inference_res.logits 
            logits = logits[:, -1, :] 
            # input_ids.shape = [1,4] # tensor([[2, 224, 609, 1590]])
            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty
            
            if temperature == 0.0: 
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  
                if top_k is not None:  
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf') 
                
                probs = F.softmax(logits, dim=-1)  
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

            if idx_next == eos:  
                break
            
            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, s:]  

        if not stream:  
            yield input_ids[:, s:]  

if __name__ == '__main__':
    config = Config()
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/train_qwen3_next/tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/train_qwen3_next/result_pretrain', 
                            num_train_epochs=1, 
                            do_train=True, 
                            per_device_train_batch_size=16,
                            gradient_accumulation_steps=2,
                            # max_steps=15000,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_total_limit=5,
                            save_steps=500,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False)          
    dataset = LLMDataset('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/train_qwen3_next/data/pretrain.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    # trainer.train(resume_from_checkpoint=True)
    trainer.save_model('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/train_qwen3_next/saves/pretrain')
    trainer.save_state()
