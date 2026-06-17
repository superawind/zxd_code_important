from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from train import LLM, Config

def infer(model_path, input_data, tokenizer):
    AutoConfig.register('small_model', Config)
    AutoModelForCausalLM.register(Config, LLM)
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    for token in model.generate({'input_ids': torch.tensor(input_data).unsqueeze(0), 'labels': None }, tokenizer.eos_token_id, 100, stream=False, temperature=0.0, top_k=8):
        print(tokenizer.decode(token[0]))
        

if __name__ == '__main__':
    # model_path = '/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/model'
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # input_data_pretrain = [tokenizer.bos_token_id] + tokenizer.encode('计算1+1')
    # infer(model_path, input_data_pretrain, tokenizer)
    
    model_path = '/mnt/code/zhaoxudong03/RL/verl_base_zxd/01_llm_releate/save_models/llm_test/sft/checkpoint-78130'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_data_sft = tokenizer.apply_chat_template([{'role':'user', 'content':'计算1+1等于多少？'}])
    infer(model_path, input_data_sft, tokenizer)
    