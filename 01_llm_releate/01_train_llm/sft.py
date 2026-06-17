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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config

if __name__ == '__main__':
    AutoConfig.register('small_model', Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/model')
    print(f'模型参数量为：：：{sum([p.numel() for p in model.parameters() if p.requires_grad])}')
    
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/model', use_fast=True)
    args = TrainingArguments(output_dir='/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/sft', 
                            num_train_epochs=5, 
                            do_train=True, 
                            per_device_train_batch_size=4,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_total_limit=5,
                            save_steps=500,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False)  
    dataset =  SFTDataset('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/data/sft_1024.jsonl', tokenizer=tokenizer, max_seq_len=1024)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/sft')
    trainer.save_state()