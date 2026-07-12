import json, threading
from pathlib import Path
import concurrent.futures
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from tqdm import tqdm
from pydantic import BaseModel
from models.models import APIModel, LitellmModel, AnthropicModel, OpenAIModel, GeminiModel
from models.basellm import BaseLLM
from statics.constants import GPTAGENTBASEURL, CLABASEURL, LITEBASEURL
from statics.attack_configs import ATTACK_CONFIGS
from utils.attack_memory import build_guided_system_prompt

class ModelFactory:
    """
    Factory class for creating different LLM model instances
    Centralizes model creation logic and makes it easy to add new model types
    """
    
    @staticmethod
    def create_model(model_name: str, system_prompt: Optional[str] = None, base_url: Optional[str] = None) -> BaseLLM:
        """
        Create a model instance based on model name
        
        Args:
            model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-sonnet')
            system_prompt: Optional system prompt for the model
            base_url: Optional custom base URL for API endpoint
            
        Returns:
            BaseLLM: An instance of the appropriate model class
            
        Examples:
            >>> model = ModelFactory.create_model('gpt-4o', 'You are a helpful assistant')
            >>> model = ModelFactory.create_model('claude-3-sonnet-20240229')
        """
        if 'claude' in model_name:
            base_url = base_url or CLABASEURL
            return AnthropicModel(model_name=model_name, system_prompt=system_prompt, base_url=base_url)
        
        # elif 'gemini' in model_name:
        #     base_url = base_url or GEMINIHTTPOPT
        #     return GeminiModel(model_name=model_name, system_prompt=system_prompt, base_url=base_url)
        
        elif 'gpt' in model_name:
            base_url = base_url or GPTAGENTBASEURL
            return OpenAIModel(model_name=model_name, system_prompt=system_prompt, base_url=base_url)
        
        elif 'llama' in model_name or 'qwen' in model_name:
            base_url = LITEBASEURL
            return LitellmModel(model_name=model_name, system_prompt=system_prompt, base_url=base_url)
        
        else:
            # Default fallback to generic API model
            base_url = base_url or GPTAGENTBASEURL
            return APIModel(model_name=model_name, system_prompt=system_prompt, base_url=base_url, max_new_tokens=8192)
    
    @staticmethod
    def get_supported_models() -> List[str]:
        """
        Get list of supported model prefixes
        
        Returns:
            List of supported model name patterns
        """
        return ['claude', 'gpt', 'hosted_vllm']
    
    @staticmethod
    def is_supported(model_name: str) -> bool:
        """
        Check if a model name is supported
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is supported, False otherwise
        """
        supported = ModelFactory.get_supported_models()
        return any(prefix in model_name for prefix in supported)


# --- 注册器机制 ---
ATTACK_REGISTRY = {}

def register_attack(name: str = None):
    """
    装饰器：用于注册 Attack 子类。
    
    Args:
        name: 注册名称。如果不填，默认使用类名。
        
    Usage:
        @register_attack("MySpecialAttack")
        class MySpecialAttack(Attack):
            ...
    """
    def decorator(cls):
        registry_name = name if name else cls.__name__
        ATTACK_REGISTRY[registry_name] = cls
        return cls
    return decorator


def register_attack_config(method, language, attack_type, config):
    # 修改：只使用字符串作为 key
    key = (method, language, attack_type) 
    # 或者拼接字符串： key = f"{method}_{language}_{attack_type}"
    
    ATTACK_CONFIGS[key] = config

@register_attack("Attack")
class Attack:
    """
    通用的攻击/评估执行器。
    不再需要子类，直接通过参数配置行为。
    """
    # 子类可以在此定义必须的参数列表，create_attack 会自动检查
    REQUIRED_PARAMS = [] 

    def __init__(
        self, 
        victim_llm: str = "gpt-4o", 
        base_url: Optional[str] = GPTAGENTBASEURL,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,  # 直接传入模板
        format_type: Union[BaseModel, str] = None, # 直接传入格式
        input_key: str | list[str] = 'meli_query',
        template_key: str = 'question',
        strategy_guidance: Optional[str] = None,
    ):
        self.victim_llm = victim_llm
        self.base_url = base_url
        self.strategy_guidance = strategy_guidance or ""
        self.system_prompt = build_guided_system_prompt(system_prompt, self.strategy_guidance)
        self.prompt_template = prompt_template
        self.format_type = format_type
        self.input_key = input_key
        self.template_key = template_key  # 默认模板键，可以在子类中覆盖
    
    def _preprocess(self, data: dict) -> str:
        """Preprocess the input data into attack prompt"""
        question = data.get(self.input_key, '')
        if self.prompt_template:
            # 使用类属性 TEMPLATE_KEY 进行格式化
            return self.prompt_template.format(**{self.template_key: question})
        # print(question)
        return question
    
    def _postprocess(self, response: str | dict) -> dict:
        """
        Postprocess the model response
        Can be overridden by subclasses if needed
        """
        if isinstance(response, dict):
            return response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"answer": response}
    
    def _generate(
        self,
        query: str,
        model: BaseLLM = None,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> List:
        
        model.reset_chat_history()
        prompt = self._preprocess(query)
        # print(prompt)
        try:
            if isinstance(self.format_type, type) and issubclass(self.format_type, BaseModel):
                res = model.chat_templ(prompt, self.format_type, generation_params)
            elif isinstance(self.format_type, str):
                res = model.chat(prompt, generation_params)
                res = {self.format_type: res}
            if self._postprocess:
                res = self._postprocess(res)        
            return res
        except Exception as e:
            print(f"Error during generation: {e}")
            return {"error": str(e), "input": query}
        
    def batch(
        self,
        queries: List[Any], # 保持 Any，以兼容 Dict 或 str
        save_input: bool | List[str] = False,
        generation_params: Optional[Dict[str, Any]] = None, # 允许传入生成参数
        concurrent: bool = False,
        max_workers: int = 5,
    ) -> List:
        """
        批量处理多个查询，支持并发执行。
        
        Args:
            queries: 查询列表，可以是字符串或包含输入键的字典
            save_input: 是否在结果中保存原始输入，或指定要保存的键列表
            max_workers: 并发时的最大工作线程数
            generation_params: 传递给生成方法的额外参数
            
        Returns:
            结果列表，与输入查询顺序对应
        """
        if concurrent and max_workers > 1:
            return self._batch_concurrent(
                queries=queries, 
                max_workers=max_workers, 
                save_input=save_input,
                generation_params=generation_params,
            )
        else:
            return self._batch(
                queries=queries, 
                save_input=save_input,
                generation_params=generation_params,
            )
    
    def _batch(
        self,
        queries: List[str],
        save_input = False,
        generation_params: Optional[Dict[str, Any]] = None,
    ) -> List:
        all_res = []
        model = ModelFactory.create_model(self.victim_llm, self.system_prompt, base_url=self.base_url)
        for query in tqdm(queries):
            # print(query)
            res = self._generate(
                    query = query, 
                    model = model, 
                    generation_params=generation_params
                )
            print(res)
            if save_input is True:
                res = query | res
            if isinstance(save_input, list):
                for key in save_input:
                    res[key] = query[key]
            all_res.append(res)
        return all_res

    def _batch_concurrent(
            self,
            queries: List[Any], # 保持 Any，以兼容 Dict 或 str
            max_workers: int = 10,
            save_input: bool | List[str] = False,
            generation_params: Optional[Dict[str, Any]] = None, # 允许传入生成参数
        ) -> List:
        """
        并发批处理方法：使用线程局部存储 (threading.local) 实现模型实例在线程间隔离和复用。
        """
        
        # 用于存储最终结果，保持原始顺序
        results = [None] * len(queries)
        
        # 线程局部存储对象，用于在每个线程中保存一个专属的模型实例
        thread_local = threading.local()

        def get_thread_safe_model() -> BaseLLM:
            """
            获取或创建当前线程独有的模型实例。
            """
            if not hasattr(thread_local, "model"):
                thread_local.model = ModelFactory.create_model(
                    self.victim_llm, 
                    self.system_prompt, 
                    base_url=self.base_url
                )
                # logging.debug(f"Initialized new model instance for thread {threading.get_ident()}")
            return thread_local.model

        def process_item(index: int, query: Any):
            """
            工作函数：由线程池调用，负责处理单个查询。
            """
            # 获取当前线程专属的模型实例
            local_model = get_thread_safe_model() 
            
            try:
                # 核心：调用统一接口，传入线程安全的模型实例
                res = self._generate(
                    query=query, 
                    model=local_model, 
                    generation_params=generation_params,
                )
                
                if save_input is True:
                    res = query | res
                if isinstance(save_input, list):
                    for key in save_input:
                        res[key] = query[key]
                
                return index, res
            except Exception as e:
                # 记录错误并返回错误结果，确保批处理不会中断
                # logging.error(f"Error processing query at index {index} in thread {threading.get_ident()}: {e}")
                return index, {"error": str(e), "input": query}

        # 使用 ThreadPoolExecutor 启动并发
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(process_item, i, q) for i, q in enumerate(queries)]
            
            # 使用 tqdm 跟踪已完成的任务
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(queries), desc="Concurrent LLM Generation"):
                # 接收结果，future.result() 会等待任务完成
                idx, result = future.result()
                results[idx] = result
        
        return results


def create_attack(
    method: str, 
    victim_llm: str = "gpt-4o", 
    base_url: Optional[str] = None,
    language: str = "en", 
    attack_type: str = "stem", 
    input_key: Optional[str] = None,
    **kwargs
) -> Attack:
    """
    工厂函数：根据配置创建 Attack 实例
    """
    # 1. 规范化参数
    if method == "ablation":
        attack_type = "stem"
        
    # 2. 构造查询键
    key = (method, language, attack_type)
    
    # 3. 获取配置
    # 改动：不再强制要求配置存在，允许“即时定义”或“默认回退”
    config = ATTACK_CONFIGS.get(key, {})
    # print(config)
    default_class = config.get("class", "Attack")
    # 如果配置没指定 class，且 method 名字刚好注册过（例如 @register_attack("sata")），则默认用它
    if "class" not in config and method in ATTACK_REGISTRY:
        default_class = method
        
    AttackClass = kwargs.pop("attack_class", default_class)
    
    # 如果 AttackClass 是字符串，尝试从注册表中查找
    if isinstance(AttackClass, str):
        if AttackClass in ATTACK_REGISTRY:
            AttackClass = ATTACK_REGISTRY[AttackClass]
        else:
            raise ValueError(f"Unknown attack class '{AttackClass}'. Available: {list(ATTACK_REGISTRY.keys())}")
    
    # 4. 确定 input_key
    # 优先使用 kwargs 里的，如果 kwargs 里没有或者为 None，则使用 config 里的
    # input_key = kwargs.get("input_key") 
    if input_key is None:
        input_key = config.get("input_key")
    
    if input_key is None:
        raise ValueError(f"Missing 'input_key' for method={method}. Please provide it in kwargs.")
    
    # 5. 准备初始化参数
    init_params = {
        "victim_llm": victim_llm,
        "base_url": base_url,
        "input_key": input_key,
        "system_prompt": config.get("system_prompt"),
        "prompt_template": config.get("prompt_template"),
        "format_type": config.get("format_type"),
    }
    
    # 合并额外的 kwargs
    init_params.update(kwargs)
    
    # 6. 参数完整性检查
    # 检查类定义的必需参数 (REQUIRED_PARAMS)
    required = getattr(AttackClass, "REQUIRED_PARAMS", [])
    missing = [key for key in required if key not in init_params or init_params[key] is None]
    
    if missing:
        raise ValueError(f"Missing required parameters for attack '{method}' (class {AttackClass.__name__}): {missing}")

    print(f"Creating attack instance of class '{AttackClass.__name__}' with params: {init_params}")
    return AttackClass(**init_params)
