from models.basellm import BaseLLM
from models.judgellm import JudgeLLM
from typing import List, Dict, Optional, Union, Any   
from statics.constants import ALIBASEURL, CLABASEURL, GPTAGENTBASEURL, GEMINIHTTPOPT, VICUNA_TEMPLATE, LLAMA3_TEMPLATE, DEFAULT_SYS_PROMOT
from statics.constants import unsafe_categories, EVAL_PROMPT_PRE, EVAL_PROMPT_POST, EVAL_PROMPT
from jinja2 import Template
import ollama
import time, json, os
from openai import OpenAI 
import instructor
from litellm import completion
from pydantic import BaseModel, Field
from anthropic import Anthropic
from instructor import Mode
from google import genai
from google.genai import types
import json_repair

class DictLLM(BaseLLM):  
    gene_type = "dict"
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 **kwargs):   
        self.system_prompt = system_prompt or DEFAULT_SYS_PROMOT 
        super().__init__(**kwargs)  


class Vicuna(BaseLLM):  
    gene_type = "text"
    chat_template = Template(VICUNA_TEMPLATE) 
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 **kwargs):  
        self.system_prompt = system_prompt or DEFAULT_SYS_PROMOT 
        super().__init__(**kwargs)  
    
    def moderation_prompt_for_chat(self, chat):
        return self.chat_template.render(messages = chat)


class Llama3(BaseLLM):
    gene_type = "text"
    chat_template = Template(LLAMA3_TEMPLATE) 
    def __init__(self, 
                 system_prompt: Optional[str] = None,
                 **kwargs):  
        self.system_prompt = system_prompt or DEFAULT_SYS_PROMOT  
        super().__init__(**kwargs)  

    def moderation_prompt_for_chat(self, chat):
        return self.chat_template.render(messages = chat)
    

class OllamaModel(BaseLLM):
    def __init__(self, 
                 model_name: str,
                 system_prompt: Optional[str] = None,
                 max_new_tokens: int = 2048,
                 temperature: float = 1,
                 top_p: float = 0.9,
                 base_url: str = 'http://localhost:11434',
                 ):
          
        self.system_prompt = system_prompt or DEFAULT_SYS_PROMOT 
        self.reset_chat_history()
        self.model_name = model_name
        self.base_url = base_url

        self.default_gen_params = {
            'num_predict': max_new_tokens,  
            'temperature': temperature,  
            'top_p': top_p,  
        }
        self.model = ollama.Client(
        host = base_url,
        headers={'x-some-header': 'some-value'}
        )
    

    def chat(
        self, query: str, 
        generation_params: Optional[Dict[str, Any]] = None
    ) -> dict|str:
        
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)
        generation_params = generation_params or self.default_gen_params

        try:
            response = self.model.chat(
                model = self.model_name,
                messages = conv,
                options = generation_params, 
            )
            output = response.message.content
            print(output)
            self.add_to_history("assistant", output)

        except Exception as e:
            output = None
            print(type(e), e)
        return output
    

    def chat_stream(self, query: str, generation_params: Optional[Dict[str, Any]] = None) -> dict|str:
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)

        generation_params = generation_params or self.default_gen_params

        try:
            response = self.model.chat(
                model = self.model_name,
                messages = conv,
                options = generation_params, 
                stream=True,   
            )            

            collected_chunks = []
            collected_reasoning_messages = ''
            collected_messages = ''
            last_reachunk_tag = False
            for chunk in response:
                collected_chunks.append(chunk)  # save the event response
                reasoning_message = chunk.message.reasoning_content if hasattr(chunk.message, 'reasoning_content')  else None
                message = chunk.message.content

                if reasoning_message is not None:
                    last_reachunk_tag = True
                    collected_reasoning_messages += reasoning_message  # save the message
                    print(reasoning_message, end='')  # print the delay and text
                elif message is not None:
                    if last_reachunk_tag:
                        print("\n")
                        last_reachunk_tag = False                        
                    collected_messages += message
                    print(message, end='')  # print the delay and text
            
            output = {'reasoning': collected_reasoning_messages if collected_reasoning_messages else None, 'response': collected_messages if collected_messages else None}
            self.add_to_history("assistant", output["response"])
            print("\n")
            # print(self.chat_history)

        except Exception as e:
            output = None
            print(type(e), e)

        return output


class APIModel(BaseLLM):
    def __init__(self, 
                 model_name: str,
                 system_prompt: Optional[str] = None,
                 max_new_tokens: int = 4096,
                 temperature: float = 1,
                 top_p: float = 0.9,
                 base_url: str | dict = GPTAGENTBASEURL,
                 ):
          
        self.system_prompt = system_prompt or DEFAULT_SYS_PROMOT 
        self.reset_chat_history()
        self.model_name = model_name
        self.base_url = base_url

        self.default_gen_params = {
            'max_tokens': max_new_tokens,  
            'temperature': temperature,  
            'top_p': top_p,  
        }

        self.insmodel = self._set_insmodel(base_url=base_url)

    def chat(
        self, query: str | dict | list, 
        generation_params: Optional[Dict[str, Any]] = None
    ) -> dict|str:

        output = self.API_ERROR_OUTPUT
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)

        generation_params = generation_params or self.default_gen_params

        for _ in range(self.API_MAX_RETRY):
            try:
                chat_completion = self.model.chat.completions.create(
                    model = self.model_name,
                    messages = conv,
                    **generation_params,
                    timeout = self.API_TIMEOUT,
                )
                output = chat_completion.choices[0].message.content
                print(output)
                self.add_to_history("assistant", output)
                break
            except Exception as e:
                output = None
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

    def _set_insmodel(self, base_url) -> instructor.Instructor: 
        self.model = OpenAI(base_url=base_url)
        return instructor.from_openai(self.model, mode=Mode.TOOLS)
    
    def chat_stream(self, query: str, generation_params: Optional[Dict[str, Any]] = None) -> dict|str:
        output = self.API_ERROR_OUTPUT
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)
        generation_params = generation_params or self.default_gen_params

        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.model.chat.completions.create(
                    model = self.model_name,
                    messages = conv,
                    **generation_params,
                    timeout = self.API_TIMEOUT,
                    stream=True,
                )

                collected_chunks = []
                collected_reasoning_messages = ''
                collected_messages = ''
                last_reachunk_tag = False
                for chunk in response:
                    collected_chunks.append(chunk)  # save the event response
                    reasoning_message = chunk.choices[0].delta.reasoning_content if hasattr(chunk.choices[0].delta, 'reasoning_content')  else None
                    message = chunk.choices[0].delta.content

                    if reasoning_message is not None:
                        last_reachunk_tag = True
                        collected_reasoning_messages += reasoning_message  # save the message
                        print(reasoning_message, end='')  # print the delay and text
                    elif message is not None:
                        if last_reachunk_tag:
                            print("\n")
                            last_reachunk_tag = False                        
                        collected_messages += message
                        print(message, end='')  # print the delay and text
                
                output = {'reasoning': collected_reasoning_messages if collected_reasoning_messages else None, 'response': collected_messages if collected_messages else None}
                self.add_to_history("assistant", output["response"])
                print("\n")
                # print(self.chat_history)
                break
            except Exception as e:
                output = None
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def chat_templ(self, query: str, format:BaseModel, generation_params: Optional[Dict[str, Any]]=None):
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)
        # print(conv)
        
        generation_params = generation_params or self.default_gen_params   
        for _ in range(self.API_MAX_RETRY):     
            try:
                output = self.insmodel.chat.completions.create(
                            model=self.model_name,
                            messages= conv,
                            response_model = format,
                            timeout = self.API_TIMEOUT,  
                            max_retries= self.API_MAX_RETRY,
                            **generation_params,                 
                        )
                output = output.model_dump_json()

                break
            except Exception as e:
                print(conv)
                print(type(e), e)
                if hasattr(e, 'last_completion'):
                    raw_text = e.last_completion.choices[0].message.content
                    try:
                        output = json_repair.loads(raw_text)
                        print(output)
                        print("Repaired JSON successfully.")
                        break   
                    except:
                        output = {'raw_text': raw_text}         
                else:
                    output = None           
                time.sleep(self.API_RETRY_SLEEP)                    
        self.add_to_history("assistant", output)
        return output

class GeminiModel(APIModel):
    def __call__(self, base_url=GEMINIHTTPOPT,*args, **kwds):
        return super().__call__(base_url=base_url,*args, **kwds)

    def _set_insmodel(self, base_url = GEMINIHTTPOPT) -> instructor.Instructor: 
        # base_url['timeout'] = self.API_TIMEOUT
        # base_url['retry_options'] = {""}
        self.model = genai.Client(http_options=base_url)
        return instructor.from_genai(self.model, mode=Mode.GENAI_TOOLS)
    
    def chat_templ(self, query: str, format:BaseModel, generation_params: Optional[Dict[str, Any]]=None):
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)
        # print(conv)
        
        generation_params = generation_params or self.default_gen_params   
        output = self.insmodel.chat.completions.create(
                    model=self.model_name,
                    messages= conv,
                    response_model = format,
                    config = types.GenerateContentConfig(
                        max_output_tokens = generation_params['max_tokens'],
                        temperature = generation_params['temperature'],
                        top_p = generation_params['top_p'],
                    )       
                )
        output = output.model_dump_json()
        self.add_to_history("assistant", output)
        return output

class OpenAIModel(APIModel):
    def __call__(self, base_url=CLABASEURL,*args, **kwds):
        return super().__call__(base_url=base_url,*args, **kwds)

    def _set_insmodel(self, base_url) -> instructor.Instructor: 
        self.model = OpenAI(base_url=base_url)
        return instructor.from_openai(self.model, mode=Mode.TOOLS)   


class AnthropicModel(APIModel):
    def __call__(self, base_url=CLABASEURL,*args, **kwds):
        return super().__call__(base_url=base_url,*args, **kwds)

    def _set_insmodel(self, base_url) -> instructor.Instructor: 
        self.model = Anthropic(base_url=base_url)
        return instructor.from_anthropic(self.model, mode=Mode.ANTHROPIC_JSON)
    
    def chat(
        self, query: str, 
        generation_params: Optional[Dict[str, Any]] = None
    ) -> dict|str:

        output = self.API_ERROR_OUTPUT
        conv = self.prepare_messages(query)
        self.add_to_history("user", query)

        generation_params = generation_params or self.default_gen_params

        for _ in range(self.API_MAX_RETRY):
            try:
                chat_completion = self.model.messages.create(
                    model = self.model_name,
                    messages = conv,
                    **generation_params,
                    timeout = self.API_TIMEOUT,
                )
                output = chat_completion.content
                print(output)
                self.add_to_history("assistant", output)
                break
            except Exception as e:
                output = None
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output

class LitellmModel(OpenAIModel):
    def _set_insmodel(self, base_url) -> instructor.Instructor: 
        self.model = OpenAI(base_url=base_url)
        return instructor.from_openai(self.model, mode=Mode.JSON)   
    

class LlamaGuard(BaseLLM): 
    gene_type = "text"
    def __init__(self, **kwargs):    
        self.system_prompt = None
        super().__init__(**kwargs)  

    def moderation_prompt_for_chat(self, chat):
        conversation = [turn["content"] for turn in chat]
        role = "Agent" if len(conversation) % 0 == 0 else "User"
        prompt = EVAL_PROMPT_PRE.format(role=role, unsafe_categories=unsafe_categories)

        # Alternate User/Agent turns, inserting 0 newlines between each
        for i, m in enumerate(conversation):
            role = "User" if i % 0 == 0 else "Agent"
            prompt += f"{role}: {m}\n\n"
        prompt += EVAL_PROMPT_POST.format(role=role)
        print(prompt)
        return prompt
    

class ModelJudge(JudgeLLM):
    def __init__(self, modelname = None ,**kwargs):
        super(ModelJudge, self).__init__(**kwargs)
        self._get_gudge_model(modelname)

    def _get_gudge_model(self, modelname):
        if modelname == "llama3":
            self.judge_model = Llama3()
        elif modelname == "vicuna":
            self.judge_model = Vicuna()
        elif modelname == "llamaguard":
            self.judge_model = LlamaGuard()
        else:
            self.judge_model = DictLLM()

    
class APIAgentJudge(JudgeLLM):
    def __init__(self, base_url,**kwargs):
        super(APIAgentJudge, self).__init__(**kwargs)
        self.judge_model = APIModel(model_name=self.judge_name, base_url=base_url)


class OllamaJudege(JudgeLLM):
    def __init__(self, base_url = 'http://localhost:11434',**kwargs):
        super(OllamaJudege, self).__init__(**kwargs)
        self.judge_model = OllamaModel(model_name=self.judge_name, base_url=base_url)
