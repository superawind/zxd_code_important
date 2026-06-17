from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import pandas as pd
import os
import time
import torch

"""
# torchrun --nproc_per_node=2 /code/zhaoxudong03/Train/Study/DP/pytorch_ddp_demo.py 
# 
1、ddp是多进程，每一个进程都加载了数据和模型，不需要通过通信传递模型和数据，每一个进程用到哪些数据，使用 DistributedSampler 实现，用在 DataLoader 创建中，注意不能和 shuffle一起使用，这样不同的进程取的数据就不同（根据rankid 来取的）
# 指定 DistributedSampler 后还有个问题，每个轮次的训练 数据顺序都一样，实际上每个 epocH 后打乱更好，且shuffle 可以实现这个能力，现在需要手动进行shuffle，使用 trainloader.sampler.set_epoch(epoch)
2、DDP 的启动需要使用torchrun ， 其中对于 dp_trainer.py 和 ddp_trainer.py 二者代码其实基本是一样的（dataset数据切分是， ddp 需要 注意设置，防止不同进程间训练集和 验证集交叉），前者用 python dp_trainer.py启动是 dp ，后者用  torchrun --nproc_per_node=2 ddp_trainer.py 启动是 ddp
3、划分数据集，必须固定随机种子，保证不同进程之间训练和验证集存在冲突的问题，采用
4、增加设置为 DDP，注意 DDP 数据分布如下： trainloader 中设置的 batch_size == 8， 两张卡情况下， 每张卡前向传播 batch_size = 8，实际BATCH_size = 16 和 DP不同，DP 是单进程，多线程，trainloader 中设置 batch_szie = 8，每张卡 batch_szie =4 

# 总结：
1、如果在训练进程内对数据集进行划分，注意保证数据划分的一致性，可以通过设置随机种子控制
2、分布式采样器会为了保证每个进程内数据大小一致，做额外的填充，评估指标可能存在偏差（accelerate 中有解决）
3、建分布式的diamante看做单进程的代码即可，只是需要分布式的数据采样器以及启动使用torchrun --nproc_per_node=2 ...
4、数据放置到指定设备上，需要正确使用 device_id，一般用local_rank， rank(即global_rank) 唯一标识，一般用来打印信息

"""


#step1、初始化进程组
dist.init_process_group(backend="nccl")

model_path = '/opt/users/Qwen3-0.6B/Qwen/Qwen3-0.6B'
data = pd.read_csv("/mnt//code/zhaoxudong03/Train/Study/DP/ChnSentiCorp_htl_all.csv")
data = data.dropna()
data

class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("/mnt//code/zhaoxudong03/Train/Study/DP/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)

dataset = MyDataset()
for i in range(5):
    print(dataset[i])

# 保证不同进程间的 数据集划分 一样，不同进程间不存在训练集和验证集交叉的问题， 设置 generator=torch.Generator().manual_seed(12)
trainset, validset = random_split(dataset, lengths=[0.5, 0.5], generator=torch.Generator().manual_seed(42))
print(len(trainset), len(validset))

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
trainloader = DataLoader(trainset, batch_size=8, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))


model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.config.pad_token_id = tokenizer.pad_token_id

# step2、如何知道两张卡，当前用哪张GPU ，一般用 LOCAL_RANK 来判断用哪张卡，和当前节点内的GPU 排序一致，
if torch.cuda.is_available():
    # model = model.cuda()
    model = model.to(int(os.environ['LOCAL_RANK']))

# step2、增加设置为 DDP，注意 DDP 数据分布如下： trainloader 中设置的 batch_size == 8， 两张卡情况下， 每张卡前向传播 batch_size = 8，实际BATCH_size = 16 和 DP不同
model = DDP(model)
optimizer = Adam(model.parameters(), lr=2e-5)

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    # step4 acc_num 是每个进程里面 正确的数量，需要汇总所有进程的数量 / 总量，有点疑惑，每个进程不是都有全部数据吗，为什么需要汇总不同进程的数据
    dist.all_reduce(acc_num)  # 默认OP 是 SUM
    return acc_num / len(validset)

def print_rank_0(info):
    if int(os.environ['RANK']) == 0:
        print(info)

def train(epoch=3, log_step=10):
    global_step = 0
    for ep in range(epoch):
        model.train()
        # step5 、指定 DistributedSampler 后还有个问题，每个轮次的训练 数据顺序都一样，实际上每个 epocH 后打乱更好，且shuffle 可以实现这个能力，现在需要手动进行shuffle，使用 trainloader.sampler.set_epoch(ep)， 每一轮都会打乱一次
        trainloader.sampler.set_epoch(ep)
        start = time.time()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ['LOCAL_RANK'])) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                # step3、添加，只有某个进程id打印loss，根据 global_rank(rank) 来确定，并打印汇总后的 loss，采用dist.all_reduce()实现
                # print(loss.item())
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                # print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        # print(f"ep: {ep}, acc: {acc}, time: {time.time() - start}")
        print_rank_0(f"ep: {ep}, acc: {acc}, time: {time.time() - start}")

train()