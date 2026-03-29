"""
模型加载和推理模块
支持基础模型 + LoRA 适配器
"""
import os
import torch
from typing import Optional
from threading import Lock

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


class SFTModel:
    """SFT模型推理类，加载基础模型+LoRA适配器"""

    _instance: Optional["SFTModel"] = None
    _lock = Lock()

    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.model = None
        self.tokenizer = None

    @classmethod
    def get_instance(
        cls,
        base_model_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ) -> "SFTModel":
        """单例模式，确保模型只加载一次"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if base_model_path is None:
                        raise ValueError("Must provide base_model_path on first initialization")
                    cls._instance = cls(
                        base_model_path=base_model_path,
                        lora_adapter_path=lora_adapter_path,
                        device=device,
                        torch_dtype=torch_dtype,
                    )
                    cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> None:
        """加载模型和分词器"""
        print(f"[SFTModel] Loading tokenizer from {self.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[SFTModel] Loading base model from {self.base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )

        if self.lora_adapter_path and os.path.exists(self.lora_adapter_path):
            print(f"[SFTModel] Loading LoRA adapter from {self.lora_adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_adapter_path,
            )
            self.model.eval()
            print(f"[SFTModel] LoRA adapter loaded successfully")
        else:
            print(f"[SFTModel] No LoRA adapter provided or path not found, using base model only")

        print(f"[SFTModel] Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        timeout: int = 120,
    ) -> str:
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # 提取 assistant 回复部分
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        return response.strip()

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> dict:
        """
        Chat格式推理，输入 messages 格式:
        [{"role": "system"|"user"|"assistant", "content": "..."}]

        返回 {"role": "assistant", "content": "..."}
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        # 构建 prompt（使用 Qwen chat template）
        prompt = self._build_chat_prompt(messages)
        response_text = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return {
            "role": "assistant",
            "content": response_text,
        }

    def _build_chat_prompt(self, messages: list[dict]) -> str:
        """构建 chat prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    @classmethod
    def reset(cls) -> None:
        """重置单例，用于切换模型或adapter"""
        with cls._lock:
            if cls._instance is not None:
                del cls._instance.model
                del cls._instance.tokenizer
                cls._instance = None
                torch.cuda.empty_cache()


def load_sft_model(
    base_model_path: str,
    lora_adapter_path: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
) -> SFTModel:
    """便捷函数：获取或创建SFT模型单例"""
    return SFTModel.get_instance(
        base_model_path=base_model_path,
        lora_adapter_path=lora_adapter_path,
        device=device,
        torch_dtype=torch_dtype,
    )
