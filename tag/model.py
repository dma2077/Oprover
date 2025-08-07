import json
import os
import time
from typing import Dict, Any, Optional, Callable
from openai import OpenAI
from tag_common import TagCommon


class APIModel:
    """通用API模型，负责调用API和输出内容，可以处理不同类型的任务"""
    
    def __init__(self, api_key: str, base_url: str, model_endpoint: str):
        """
        初始化模型
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model_endpoint: 模型端点
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_endpoint = model_endpoint
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    
    def call_api_sync(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """
        同步调用API
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            API响应内容
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_endpoint,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"API调用失败: {e}")
    
    def predict(self, prompt: str, expected_keys: Optional[list] = None, 
                validate_fn: Optional[Callable] = None, max_retries: int = 3, 
                delay: float = 1.5) -> Dict[str, Any]:
        """
        通用预测方法
        
        Args:
            prompt: 输入提示
            expected_keys: 期望的JSON键
            validate_fn: 验证函数
            max_retries: 最大重试次数
            delay: 重试延迟
            
        Returns:
            预测结果字典
        """
        for attempt in range(max_retries):
            try:
                response = self.call_api_sync(prompt)
                parsed = TagCommon.extract_json_from_response(response, expected_keys=expected_keys)
                data = json.loads(parsed)
                
                # 如果提供了验证函数，进行验证
                if validate_fn and not validate_fn(data):
                    raise ValueError("Validation failed")
                
                return {
                    "data": data,
                    "response": response,
                    "status": "success"
                }
                
            except Exception as e:
                print(f"[Prediction Retry {attempt + 1}] Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    return {
                        "data": None,
                        "response": None,
                        "status": "failed",
                        "error": str(e)
                    }
    
    def predict_with_retry(self, func: Callable, max_retries: int = 3, delay: float = 1.5) -> Any:
        """
        带重试机制的预测
        
        Args:
            func: 要执行的函数
            max_retries: 最大重试次数
            delay: 重试延迟
            
        Returns:
            函数执行结果
        """
        return TagCommon.retry_with_backoff(func, max_retries, delay)


# 默认配置
DEFAULT_API_KEY = os.environ.get("ARK_API_KEY", "92730715-072f-4393-89e8-5ac2a2c43348")
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL_ENDPOINT = "doubao-1-5-pro-32k-250115"

# 创建默认模型实例
default_model = APIModel(DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL_ENDPOINT)