# src/mcp_llm_bridge/config.py
from dataclasses import dataclass
from typing import Optional
from mcp import StdioServerParameters

@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000

@dataclass
class BridgeConfig:
    """Configuration for the MCP-LLM Bridge"""
    mcp_server_params: StdioServerParameters
    llm_config: LLMConfig  # Function Calling用の設定
    thinking_config: Optional[LLMConfig] = None  # 思考プロセス用の設定
    system_prompt: Optional[str] = None
    
    def get_thinking_config(self) -> LLMConfig:
        """思考プロセス用の設定を取得（デフォルトはllm_configを使用）"""
        if self.thinking_config:
            return self.thinking_config
        return self.llm_config
