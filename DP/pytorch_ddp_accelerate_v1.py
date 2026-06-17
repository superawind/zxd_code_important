from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import time
import pandas as pd

from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


"""
# 启动方式一 ：torchrun --nproc_per_node=2 pytorch_ddp_accelerate.py ， 没有accelerate 启动 兼容的方式多
# 启动方式二： accelerate launch pytorch_ddp_accelerate.py 最基础的方式
    可以通过终端输入 accelerate config 开始配置启动配置文件
    accelerate launch --help 查看参数提示信息


1、不需要手动初始化进程组，dist.init_progress_group()， accelerate 初始化对象的时候会执行这句
2、DataLoader 的构造，和单卡一样，采用shuffle就行，不需要采用 DistributedSampler
3、不需要手动将模型等移动到某个设备上，普通初始化模型和 optim 就行

"""

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("/mnt//code/zhaoxudong03/Train/Study/DP/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)

def prepare_dataloader():
    dataset = MyDataset()

    # 保证不同进程间的 数据集划分 一样，不同进程间不存在训练集和验证集交叉的问题， 设置 generator=torch.Generator().manual_seed(12)
    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def collate_func(batch):
        # batch 中每一个元素。是 dataset.getitem() 的一条返回结果
        texts, labels = [], []
        # print(len(batch), len(batch[0]))
        # print('========================================')
        # print(batch)
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    # step1
    trainloader = DataLoader(trainset, batch_size=8, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader

def prepare_model_and_optimizer():
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = Adam(model.parameters(), lr=2e-5)

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            # 这一步 accelerate 会自动进行 汇总聚合，不需要手动汇总，这一步解决了之前 ddp 的一个问题，验证数据，batch_size = 64，数据集不够平均划分的时候，ddp 那种会自动填充到 64个，会导致acc 有点虚高，甚至超过 1, 采用 accelerator.gather_for_metric()可以解决这个问题。最后一个batch size 是实际大小
            preds, refs = accelerator.gather_for_metrics((pred, batch['labels']))
            acc_num += (preds.long() == refs.long()).float().sum()

    return acc_num / len(validloader.dataset)

# 也不需要自己定义 rank 0 打印
def print_rank_0(info):
    if int(os.environ['RANK']) == 0:
        print(info)

def train(model, optimizer, trainloader, validloader, accelerator:Accelerator, epoch=3, log_step=10):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            accelerator.backward(loss)
            optimizer.step()
            if global_step % log_step == 0:
                loss = accelerator.reduce(loss, 'mean')
                accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate(model, validloader, accelerator)
        # print(f"ep: {ep}, acc: {acc}, time: {time.time() - start}")
        accelerator.print(f"ep: {ep}, acc: {acc} ")

def main():
    accelerator = Accelerator()
    trainloader, validloader = prepare_dataloader()
    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)
    train(model, optimizer, trainloader, validloader, accelerator)

if __name__ == '__main__':
    model_path = '/opt/users/Qwen3-0.6B/Qwen/Qwen3-0.6B'

    main()