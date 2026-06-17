import logging
import os
import time
import warnings

import ray
import torch

warnings.filterwarnings("ignore")

from typing import List, Tuple

from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402
from verl.single_controller.ray.base import (  # noqa: E402
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.device import (  # noqa: E402
    get_device_name,
    get_nccl_backend,
)

ray.init()

def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    world_size = worker_group.world_size

    # Process args - split each arg list across workers
    split_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            # Split the list into world_size chunks
            chunks = [[] for _ in range(world_size)]
            for i, item in enumerate(arg):
                chunks[i % world_size].append(item)
            split_args.append(tuple(chunks))
        else:
            # If not a list/tuple, duplicate it for each worker
            split_args.append(tuple([arg] * world_size))

    # Process kwargs - split each kwarg list across workers
    split_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple)):
            # Split the list into world_size chunks
            chunks = [[] for _ in range(world_size)]
            for i, item in enumerate(v):
                chunks[i % world_size].append(item)
            split_kwargs[k] = tuple(chunks)
        else:
            # If not a list/tuple, duplicate it for each worker
            split_kwargs[k] = tuple([v] * world_size)

    return tuple(split_args), split_kwargs


# def dispatch_dp_compute():
#     pass

def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Check that we have output from each worker
    assert len(output) == worker_group.world_size

    merged_output = []
    for worker_output in output:
        if isinstance(worker_output, list):
            merged_output.extend(worker_output)
        else:
            merged_output.append(worker_output)
    
    return merged_output

def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


device_name = get_device_name()

@ray.remote
class TestAccelerateWorker(Worker):
    def __init__(self):
        super().__init__()
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # 下面这句本质上等价于 torch.distributed.init_progress_group("gloo", rank=rank, init_method=init_method, world_size=world_size)
        self.distributed_state = PartialState(
            backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
            rank=rank,
            world_size=world_size,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def show_info(self):
        info = {
            "acc_device": self.distributed_state.device,
            "rank": self.rank,
            "world_size": self.world_size,
            "acc_world_size": str(self.distributed_state),
        }
        return info
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_model(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = self.distributed_state.device,
            torch_dtype = torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model.device
    
    def _infer(self, prompts: list[str]):
        def format_prompt_func(prompt: str):
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            return text
        
        res = []
        for batch_simple in prompts:
            batch = format_prompt_func(batch_simple)
            model_inputs = self.tokenizer([batch], return_tensors='pt').to(self.distributed_state.device)
            
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
    
            response = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0][:10]
            
            res.append({'query':batch_simple, 'response': response})
        
        return res
    
    # way1
    @register(dispatch_mode = Dispatch.ONE_TO_ALL)
    def infer_way1(self, prompt: str|list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
            
        return self._infer(prompts)


    @register(dispatch_mode = Dispatch.ONE_TO_ALL)
    def infer_way2(self, prompt: str|list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
            
        splits = [[] for _ in range(self.world_size)]
        for i, prompt in enumerate(prompts):
            splits[ i % self.world_size].append(prompt)
            

        split_world_prompts = splits[self.rank]
        return self._infer(split_world_prompts)
    
    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_one_to_all,
            "collect_fn": collect_dp_compute,
        }
    )
    def infer_way3(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
            
        splits = [[] for _ in range(self.world_size)]
        for i, prompt in enumerate(prompts):
            splits[ i % self.world_size].append(prompt)
            

        split_world_prompts = splits[self.rank]
        return self._infer(split_world_prompts)
    
    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_one_to_all,
            "collect_fn": collect_dp_compute,
        }
    )
    def infer_way3_1(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = prompts[self.rank]

        res = self._infer(prompts)
        return res

    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_dp_compute,
            "collect_fn": collect_dp_compute,
        }
    )
    def infer_way4(self, prompt: str | list[str]):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = prompts[self.rank]

        res = self._infer(prompts)
        return res


# 初始化资源
resource_pool = RayResourcePool([2], use_gpu=True)

# 类初始化、工作组 初始化
class_with_args = RayClassWithInitArgs(cls=TestAccelerateWorker)
worker_group = RayWorkerGroup(resource_pool, class_with_args)

show_info = worker_group.show_info()

for i in show_info:
    print(i)


# show_info = worker_group.show_info()

# for i in show_info:
#     print(i)


query_list = [
    "你是谁",
    "1+1=几",
    "十个字介绍一下杭州",
]


model_name = "/opt/users/models/Qwen3-4B"
model_device = worker_group.load_model(model_name=model_name)

# way 1
# 打印结果如下，所有的 rank 都会推理全部数据，
# [[{'query': '你是谁', 'response': '<think>\n好的'}, {'query': '1+1=几', 'response': '<think>\n嗯，'}, {'query': '十个字介绍一下杭州', 'response': '<think>\n好的'}], [{'query': '你是谁', 'response': '<think>\n好的'}, {'query': '1+1=几', 'response': '<think>\n嗯，'}, {'query': '十个字介绍一下杭州', 'response': '<think>\n好的'}]]
response_list = worker_group.infer_way1(prompt=query_list)
print(response_list)

print('===============================================way2 begin ======================================================')
# way2 按照 world_size 分发, 但是结果是分开的，不是一个统一的list
# [[{'query': '你是谁', 'response': '<think>\n好的'}, {'query': '十个字介绍一下杭州', 'response': '<think>\n好的'}], [{'query': '1+1=几', 'response': '<think>\n嗯，'}]]
response_list = worker_group.infer_way2(prompt=query_list)
# print(response_list)
for i in response_list:
    print(i)

print('===============================================way3 begin ======================================================')
# 正确解决问题
# [{'query': '你', 'response': '<think>\n好的'}, {'query': '是', 'response': '<think>\n好的'}, {'query': '谁', 'response': '<think>\n好的'}, {'query': '1', 'response': '<think>\nOk'}, {'query': '+', 'response': '<think>\nOk'}, {'query': '1', 'response': '<think>\nOk'}, {'query': '=', 'response': '<think>\nOk'}, {'query': '几', 'response': '<think>\n好的'}]
response_list = worker_group.infer_way3(prompt=query_list)
print(response_list)

print('===============================================way3_1 begin ======================================================')
# 采用默认的分发函数，存在如下问题，每个 rank 只取一条，且分配的函数将一条数据给便利拆分了，汇总函数实现了将结果同意在一个list 中
# [{'query': '你', 'response': '<think>\n好的'}, {'query': '是', 'response': '<think>\n好的'}, {'query': '谁', 'response': '<think>\n好的'}, {'query': '1', 'response': '<think>\nOk'}, {'query': '+', 'response': '<think>\nOk'}, {'query': '1', 'response': '<think>\nOk'}, {'query': '=', 'response': '<think>\nOk'}, {'query': '几', 'response': '<think>\n好的'}]
response_list = worker_group.infer_way3_1(prompt=query_list)
print(response_list)


print('===============================================way4 begin ======================================================')
# 满足需求， 不同节点分发数据，结果汇总在一个 list 中
# [{'query': '你是谁', 'response': '<think>\n好的'}, {'query': '十个字介绍一下杭州', 'response': '<think>\n好的'}, {'query': '1+1=几', 'response': '<think>\n嗯，'}]
response_list = worker_group.infer_way4(prompt=query_list)
print(response_list)


time.sleep(30)

ray.shutdown()
