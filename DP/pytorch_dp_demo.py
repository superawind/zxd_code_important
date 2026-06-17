from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.optim import Adam
import pandas as pd
import time
import torch


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

trainset, validset = random_split(dataset, lengths=[0.5, 0.5])
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


trainloader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)


model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.config.pad_token_id = tokenizer.pad_token_id


if torch.cuda.is_available():
    model = model.cuda()

# 增加设置为 DP，注意 DP 数据分布如下： trainloader 中设置的 batch_size == 8， 两张卡情况下， 每张卡前向传播 batch_size = 4
model = DataParallel(model)
print('model.device_ids:::', model.device_ids) # 双卡结果为 [0, 1]

optimizer = Adam(model.parameters(), lr=2e-5)


def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)

def train(epoch=3, log_step=10):
    global_step = 0
    for ep in range(epoch):
        model.train()
        start = time.time()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            # 多卡 DP 此时 loss 是 多卡 loss 拼接后的结果，元素数量等于DP 数的列表，因此需要汇总一下
            loss = output.loss.mean()
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}, time: {time.time() - start}")

# 1、训练验证
# train()

# 增加设置为 DP，注意 DP 数据分布如下： trainloader 中设置的 batch_size == 8， 两张卡情况下， 每张卡前向传播 batch_size = 4
# DP 的启动，直接使用 python ...py 文件即可
# 训练过成不建议使用 DP，因为 负载不均衡问题， 单进程多线程 GIL锁问题，只能单机多卡问题，导致效果不佳，其中耗时上，模型太大，多用来从 serve 复制到 work ，耗时严重，但是推理过程，不涉及权重更新，不需要拷贝模型，因此推理可以使用，不过得修改 nn.DataParallel里的代码
# DataParallel没有 generate 方法，只能forward 生成 logits

# 2、推理验证， 
# 2.1、单 gpu 推理
def test_one_gpu():
    # model 经过 nn.DataParallel 封装后，可以通过 model.module 取出来真正的模型
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model.module(**batch) # 获取真正的模型进行推理

# start = time.time()
# test_one_gpu()
# print(time.time()-start) # 15.433281898498535   3882 条数据


# 2.2、双 gpu 推理
def test_two_gpu():
    # model 经过 nn.DataParallel 封装后，可以通过 model.module 取出来真正的模型
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch) # 获取真正的模型进行推理

# start = time.time()
# test_two_gpu()
# print(time.time()-start) # 16.658422470092773   3882 条数据 ，比单卡还慢，因为拷贝模型原因导致的主要是


# 2.3、自定义 DP，因为只推理，所以直接将 nn.DataParallel 中的模型拷贝部分(只在最开始拷贝一次)，值进行前向计算，不拷贝模型
# 双 gpu 推理
replicas = model.replicate(model.module, model.device_ids[:2]) # 模型拷贝部分，放在外面，只拷贝一次, 两张卡改成 2 即可
def test_new_dp():
    # model 经过 nn.DataParallel 封装后，可以通过 model.module 取出来真正的模型
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            # 分发数据
            inputs, module_kwargs = model.scatter(inputs=None, kwargs=batch, device_ids = model.device_ids)
            outputs = model.parallel_apply(replicas, inputs, module_kwargs)
            outputs = model.gather(outputs, model.output_device)

            
start = time.time()
test_new_dp()
print(time.time()-start) # 10.453891038894653   3882 条数据 ，快于单卡



# nn.DataParallel 的 forward 源码
"""
def forward(self, *inputs: Any, **kwargs: Any) -:
    with torch.autograd.profiler.record_function("DataParallel.forward"):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                    f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                                    f"them on device: {t.device}")

        inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)
        # for forward function without any inputs, empty list and dict will be created
        # so the module can be executed on one device which is the first one in device_ids
        if not inputs and not module_kwargs:
            inputs = ((),)
            module_kwargs = ({},)

        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **module_kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, module_kwargs)
        return self.gather(outputs, self.output_device)
"""