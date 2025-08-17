from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from utils.build_conversation import build_conversation
from config.config_wrapper import config_wrapper

def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    max_model_len = model_args.get('max_model_len', None)  # ✅ 新增：从参数中获取
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_kwargs = {
            "model": model_path,
            "tokenizer": model_path,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": 0.95,
            "trust_remote_code": True,
            "disable_custom_all_reduce": True,
            "enforce_eager": True,
        }
        if max_model_len is not None:
            model_kwargs["max_model_len"] = max_model_len  # ✅ 如果有就加入
        
        model_components['model'] = LLM(**model_kwargs)
        model_components['model_name'] = model_name
       
    else:
        model_components['use_accel'] = False
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        model_components['model_name'] = model_name
    return model_components

def infer(prompts, historys=[{}], system_prompt='', **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    
    if model_name == "Kimina-Prover-72B" and not system_prompt:
        system_prompt = "You are an expert in mathematics and proving theorems in Lean 4."

    if isinstance(prompts[0], str):
        messages = [build_conversation(history, prompt, system_prompt) for history, prompt in zip(historys, prompts)]
    else:
        raise ValueError("Invalid prompts format")
    
    if use_accel:
        prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True) for message in messages]
        stop_token_ids=[tokenizer.eos_token_id]
        if 'Llama-3' in model_name:
            stop_token_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        top_p = config_wrapper.top_p if config_wrapper.top_p is not None else 1.0
        sampling_params = SamplingParams(max_tokens=config_wrapper.max_tokens, top_p=top_p, stop_token_ids=stop_token_ids, temperature=config_wrapper.temperatrue)
        outputs = model.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
        responses = []
        for output in outputs:
            response = output.outputs[0].text
            responses.append(response)
    else:
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, padding=True, truncation=True, return_dict=True, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=config_wrapper.max_tokens, do_sample=False)
        responses = []
        for i, prompt in enumerate(prompts):
            response = tokenizer.decode(outputs[i, len(inputs['input_ids'][i]):], skip_special_tokens=True)
            responses.append(response)

    return responses, [None] * len(responses)

if __name__ == '__main__':

    prompts = [
        '''Who are you?''',
        '''only answer with "I am a chatbot"''',
    ]
    model_args = {
        'model_path_or_name': '01-ai/Yi-1.5-6B-Chat',
        'model_type': 'local',
        'tp': 8
    }
    model_components = load_model("Yi-1.5-6B-Chat", model_args, use_accel=True)
    responses = infer(prompts, None, **model_components)
    for response in responses:
        print(response)
        break
