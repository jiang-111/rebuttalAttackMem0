from typing import List, Dict, Optional, Union, Any   
from vllm import LLM, SamplingParams
import emoji,re, json, time
import instructor
from pydantic import BaseModel, Field


class BaseLLM:  
    
    """  
    基础LLM类，提供模型加载、聊天和推理功能  
    可以被扩展以支持不同的模型或自定义行为  
    """  
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 1
    API_TIMEOUT = 500
        
    def __init__(  
        self,  
        model_name: str = "lmsys/vicuna-1.5-7b",  
        torch_dtype: str = 'auto',  
        system_prompt: Optional[str] = None,  
        max_new_tokens: int = 2048,  
        temperature: float = 0.7,  
        top_p: float = 0.9,  
        model_kwargs: Optional[Dict[str, Any]] = None,  
        generation_kwargs: Optional[Dict[str, Any]] = None,  
    
    ):  
        
        self.model_name = model_name  
        self.default_gen_params = {
            'max_tokens': max_new_tokens,  
            'temperature': temperature,  
            'top_p': top_p,  
        }
            
        if generation_kwargs:  
            self.default_gen_params.update(generation_kwargs)  
        
        model_kwargs = model_kwargs or {}  
        tensor_parallel_size = model_kwargs.pop("tensor_parallel_size", 1)
        gpu_memory_utilization = model_kwargs.pop("gpu_memory_utilization", 0.9)
        
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=torch_dtype,
            trust_remote_code=True,   
            **model_kwargs
        )
        self.reset_chat_history()  
        self.insmodel = self._set_insmodel()
    
    def reset_chat_history(self, systempmt = None) -> None:  
        self.chat_history = []  
        if systempmt is not None:
            self.system_prompt = systempmt
        if self.system_prompt is not None:  
            self.add_to_history("system", self.system_prompt)
    
    def add_to_history(self, role: str, query: str|dict|list) -> None:  
        if isinstance(query, str):
            self.chat_history.append({"role": role, "content": query})  
        elif isinstance(query, dict):
            self.chat_history.append(query)  
        elif isinstance(query, list):
            self.chat_history.extend(query)

    def prepare_messages(self, query: str | dict | list, role: str = 'user') -> List[Dict[str, str]]:  
        messages = self.chat_history.copy()  
        if isinstance(query, str):
            messages.append({"role": role, "content": query})
        elif isinstance(query, dict):
            messages.append(query)
        elif isinstance(query, list):
            messages.extend(query)
            
        for i in range(len(messages)):
            if "content" in messages[i]:
                try:
                    messages[i]["content"] = self.text_process(messages[i]["content"])
                except Exception as e:
                    print(e)
                    pass        
        return messages  
    
    def generate_from_messages_dict(  
        self,   
        messages: List[Dict[str, str]],  
        generation_params: Optional[Dict[str, Any]] = None  
    ) -> str:   
        gen_params = self.default_gen_params
        if generation_params:  
            gen_params.update(generation_params)
        sampling_params = SamplingParams(**gen_params)
        outputs = self.model.chat(messages, sampling_params)
        res = outputs[0].outputs[0].text.strip()
        return res        
    
    def generate_from_messages_text(  
        self,   
        messages: List[Dict[str, str]],  
        generation_params: Optional[Dict[str, Any]] = None  
    ) -> str:  
        
        gen_params = self.default_gen_params
        if generation_params:  
            gen_params.update(generation_params)
        sampling_params = SamplingParams(**gen_params)
        query = self.moderation_prompt_for_chat(chat=messages)
        outputs = self.model.generate(query, sampling_params)
        assistant_response = outputs[0].outputs[0].text.strip()

        return assistant_response    
    

    def moderation_prompt_for_chat(self, chat):
        pass


    def remove_code_blocks(self, text):
        pattern = r"```.*?```"
        return re.sub(pattern, "", text, flags=re.DOTALL)
    
    
    def text_process(self, model_response):
        model_response = emoji.replace_emoji(model_response, replace="")
        model_response = self.remove_code_blocks(model_response)

        model_response = model_response.replace("```", "")
        return model_response    
    

    def generate_from_messages(  
        self,   
        messages: List[Dict[str, str]],  
        generation_params: Optional[Dict[str, Any]] = None  
    ) -> str:  
        
        if self.gene_type == "dict":
            return self.generate_from_messages_dict(messages, generation_params)
        elif self.gene_type == "text":
            return self.generate_from_messages_text(messages, generation_params)


    def chat(  
        self,   
        query: str, 
        generation_params: Optional[Dict[str, Any]] = None  
    ) -> str:  
        
        messages = self.prepare_messages(query)  
        
        try:
            response = self.generate_from_messages(messages, generation_params)
            self.add_to_history("user", query) 
            self.add_to_history("assistant", response)  
        except Exception as e:
            print(e)
            response = e
        return response  
    
    def chat_stream(self, query: str, generation_params: Optional[Dict[str, Any]] = None) -> dict|str:
        pass
    
    def _set_insmodel(self) -> instructor.Instructor: 
        pass
    
    def delete(self):
        try:
            import gc
            del self.model
            gc.collect()
        except:
            print("no model")