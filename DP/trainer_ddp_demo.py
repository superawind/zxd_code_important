## Trainer 中的逻辑 model = _warp_model(model) 默认会检查设备，并直接走 DP

## DP 的问题在于 ，单进程多线程，GIL 锁的问题可能导致多卡无法真正发挥效率， 2、DP 每次更新完参数，都需要从SERVE 机器上将模型赋值给WORK 机器，耗时严重
## DP 的作用是可以手动改一下用于推理，因为没有更新参数，所以不进行每次模型的赋值，减少耗时，正常 DP 不一定比单卡速度快，可以尝试增加 Batch_size 



from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import torch 

dataset = load_dataset('csv', data_files=['/mnt//code/zhaoxudong03/Train/Study/DP/ChnSentiCorp_htl_all.csv'], split='train')
dataset = dataset.filter(lambda x: x['review'] is not None)
dataset

datasets = dataset.train_test_split(test_size=0.1, seed=42)  # 注意DDP 一定要加 seed=42，保证不同进程之间，训练集和 验证集不存在交叉问题
print(datasets)


model_path = '/opt/users/Qwen3-0.6B/Qwen/Qwen3-0.6B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.eos_token

print(tokenizer.eos_token, tokenizer.pad_token)

def process_function(examples):
    # print(examples) # 形状为 组成 batch 的 {'key1': [v1, v2, ...], 'key2':[v1, v2,...]}
    tokenized_examples = tokenizer(examples['review'], max_length=128, truncation=True, padding=True, return_tensors='pt')
    tokenized_examples['labels'] = examples['label']
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets['train'].column_names)
tokenized_datasets

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

print(model.config.pad_token_id, '\n\n',  model.device)
model.config.pad_token_id = tokenizer.pad_token_id
import evaluate


acc_metric = evaluate.load("/mnt/code/zhaoxudong03/Train/Study/DP/metric_accuracy.py")
f1_metirc = evaluate.load("/mnt/code/zhaoxudong03/Train/Study/DP/metric_f1.py")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

train_args = TrainingArguments(output_dir='checkpoints',
                               per_device_train_batch_size = 2,
                               per_device_eval_batch_size = 8,
                               logging_steps=10,
                               eval_strategy='epoch',
                               save_strategy='epoch',
                               save_total_limit = 3,
                               learning_rate = 2e-5,
                               weight_decay=0.01,
                               metric_for_best_model='f1',
                               load_best_model_at_end=True)

trainer = Trainer(model=model, 
                  args=train_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"], 
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()


