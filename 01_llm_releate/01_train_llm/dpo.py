from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from dataset import DPODataset, DPODataCollator
from train import LLM, Config
import os 

os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'

def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

def mask_logits(logits, labels):
    # logits shape: (batch_size, seq_len)
    # labels_masks shape: (batch_size, seq_len)
    new_logits = []
    for logit, label in zip(logits, labels): # logits.shape = [seq_len] 是一条数据
        new_logits.append(logit[label != 0].sum().unsqueeze(0))  # 计算所有非pad 位置 token 的 概率和
    return new_logits # 列表中每个元素都是一个 shape = [1] 的tensor，一个张量值，列表大小为 bs * 卡数量 * 2

def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)
    # ref_chose_probs、 ref_reject_probs、 chosen_rpobs、 reject_probs、 pi_logratios、 ref_logratios、 loss的形状为 [bs*卡数量]
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()

class DPOTrainer(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs['input_ids']  # [bs * 卡数量 * 2, seq_len]  之所以乘以2，是因为前半部分是正利，后半部分是负例
        labels = inputs['labels']        # [bs * 卡数量 * 2, seq_len]
        with torch.no_grad():
            ref_logits = ref_model(input_ids=input_ids, labels = labels).logits  # [bs * 卡数量 * 2, 511, 6400] [bs, seq_len, vocab_size]
        # [bs * 卡数 * 2, seq_len]
        ref_probs = logits_to_probs(ref_logits, labels)
        ref_probs = mask_logits(ref_probs, labels)
        logits = model(input_ids=input_ids, labels = labels).logits
        probs = logits_to_probs(logits, labels)
        probs = mask_logits(probs, labels)
        loss = dpo_loss(ref_probs, probs, 0.1) # ref_probs probs 是大小为 [bs*卡数*2] 的列表，每一个元素 是一个shape=[1]的张量值
        return loss
        

if __name__ == '__main__':
    AutoConfig.register('small_model', Config)
    AutoModelForCausalLM.register(Config, LLM)
    model = AutoModelForCausalLM.from_pretrained('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/sft/checkpoint-78130')
    ref_model = AutoModelForCausalLM.from_pretrained('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/sft/checkpoint-78130').eval().to('cuda')
    print(f'模型可训练参数量为：：： {sum([p.numel() for p in model.parameters() if p.requires_grad])}')
    
    tokenizer = AutoTokenizer.from_pretrained('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/sft/checkpoint-78130', use_fast=True)
    data_collator = DPODataCollator(tokenizer, max_seq_len=512) # 加载的大模型旋转位置编码最大长度为1024，这里不能超过这个值
    args = TrainingArguments(output_dir='/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/dpo-1-epoch', 
                            num_train_epochs=1,  # 训练太多轮，模型似乎会输出很多重复内容
                            do_train=True, 
                            per_device_train_batch_size=3,
                            gradient_accumulation_steps=4,
                            # max_steps=15000,
                            logging_steps=50,
                            report_to='tensorboard',
                            save_total_limit=3,
                            bf16=True,
                            learning_rate=0.00001,  # 学习率很重要，太大会把模型训飞
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=1,
                            dataloader_pin_memory=True,
                            save_safetensors=False,
                            save_steps=100) 
    dataset = DPODataset('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/data/dpo.json', tokenizer=tokenizer)
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/dpo-1-epoch/model')
    trainer.save_state()
   