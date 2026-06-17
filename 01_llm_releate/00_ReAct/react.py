import json 
import re
import requests
import random 
import urllib.parse
from typing import Iterable
from openai import OpenAI


# 工具定义
tools = [
    {
        "type": "function",
        "function":{
            "name": "get_weather",
            "description": "Get weather",
            "parameters":{
                "type": "object",
                "properties": {
                    "location": {'type': "string", "description": "location"}
                },
                "required": ["location"]
            }
        }
    }
]

def get_system_prompt():
    tool_strings = "\n".join([json.dumps(tool["function"]) for tool in tools])
    tool_names = ", ".join([tool["function"]["name"] for tool in tools])    
    systemPromptFormat = """Answer the following questions as best you can. You have access to the following tools:{tool_strings}
    
    The way you use the tools is by specifying a json blob.
    Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).
    
    The only values that should be in the "action" field are: {tool_names}
    
    The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:
    
    ```{{{{
        "action": $TOOL_NAME,
        "action_input": $INPUT
       }}}}
    ```
    ALWAYS use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: the result of the action
    ... (this Thought/Action/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin! Reminder to always use the exact characters `Final Answer` when responding. 
    """    

    return systemPromptFormat.format(tool_strings=tool_strings, tool_names=tool_names)

# 实现获取天气
def get_weather(location: str) -> str:    
    return random.choice(["晴天","多云","小雨","大雨","雷阵雨","阴天"])
    # url = "http://weather.cma.cn/api/autocomplete?q=" + urllib.parse.quote(location)    
    # # try:
    # response = requests.get(url)    
    # print('url response:::', response)
    # # except:
    # # url = f"http://weather.cma.cn/api/now/{location}"    
    # # response1 = requests.get(url).text
    # # print('url response111:::', response1)
        
    
    # data = response.json()    
    # if data["code"] != 0:        
    #     return "没找到该位置的信息"    
    # location_code = ""    
    # for item in data["data"]:        
    #     str_array = item.split("|")        
    #     if (str_array[1] == location or str_array[1] + "市" == location or str_array[2] == location):
    #         location_code = str_array[0] 
    #         break    
    #     if location_code == "":        
    #         return "没找到该位置的信息"    
    #     url = f"http://weather.cma.cn/api/now/{location_code}"    
    #     return requests.get(url).text
    
# 实现工具调用
def invoke_tool(toolName:  str, toolParamaters) ->  str:      
    result =  ""       
    if  toolName ==  "get_weather":            
        result = get_weather(toolParamaters["location"])       
    else:            
        result =  f"函数{toolName}未定义"       
    return  result


def main(query = "北京和广州的天气怎么样"):
    
    systemMsg = get_system_prompt()
    maxIter = 5                         # 最大迭代次数，每一次迭代调用一次模型
    agent_scratchpad = ""               # agent 思考过程
    action_parttern = re.compile(r"\nAction:\n`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL)  # 解析 Action 
    for iterSeq in range(1, maxIter+1):
        user_query = f"Question:{query}\n\n{agent_scratchpad}"
        messages = [
            {'role': 'system', 'content': systemMsg},
            {'role': 'user', 'content': user_query}
        ]
        
        print(f">> iterSeq:{iterSeq}")
        print(f">>> messages:{json.dumps(messages, ensure_ascii=False)}")
        
        # 向LLM发起请求，注意需要设置stop参数
        # chat_completion = client.chat.completions.create(messages=messages,                  
        #                                                  model=model,                  
        #                                                  stop="Observation:",           
        #                                                  )      
        chat_completion = client.chat.completions.create(
            model="default",
            messages=messages, 
            temperature=0.6,
            top_p=0.95,
            max_tokens=8192,
            extra_body={
                "top_k":20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stop=['<|im_end|>', '<|im_end|>', '<|end_of_response>', "Observation"]
        )     
         
        content = chat_completion.choices[0].message.content             
        print(f">>> content:\n{content}")
        final_answer_match = re.search(r"\nFinal Answer:\s*(.*)", content)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            print(f'>>>最终答案：{final_answer}')
            return 
        action_match = action_parttern.search(content)
        if action_match:
            obj = json.loads(action_match.group(1))
            toolName = obj['action']  
            toolParameters = obj['action_input']   
            print(f">>> tool name:{toolName}")
            print(f">>> tool parameters:{toolParameters}")
            result = invoke_tool(toolName, toolParameters)
            print(f">>> tool result: {result}")
            # 把本次LLM的输出(Though/Action)和工具调用结果(Observation)添加到agent_scratchpad            
            agent_scratchpad += content + f"\nObservation: {result}\n"
        else:
            print(">>> ERROR: detect invalid response")
            return 
          
    print(">>> 迭代次数达到上限，我无法得到最终答案")   
        
        
    
    
if __name__ == '__main__':
    # 具体流程参考 https://mp.weixin.qq.com/s/9V5T4k75-XcQq-qo06Cx3Q
    
    openai_api_key = "EMPTY"
    openai_api_base = 'http://10.16.80.9:8027/v1'
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    query = "北京和广州的天气怎么样"
    main(query)
    
    
