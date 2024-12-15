# src/mcp_llm_bridge/llm_client.py
from typing import Dict, List, Any, Optional
import openai
from mcp_llm_bridge.config import LLMConfig
import logging
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s:     %(cyan)s%(name)s%(reset)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class LLMResponse:
    """Standardized response format focusing on tool handling"""
    def __init__(self, completion: Any):
        self.completion = completion
        self.choice = completion.choices[0]
        self.message = self.choice.message
        self.stop_reason = self.choice.finish_reason
        self.is_tool_call = self.stop_reason == "tool_calls"
        
        # Format content for bridge compatibility
        self.content = self.message.content if self.message.content is not None else ""
        self.tool_calls = self.message.tool_calls if hasattr(self.message, "tool_calls") else None
        
        # Debug logging
        logger.debug(f"Raw completion: {completion}")
        logger.debug(f"Message content: {self.content}")
        logger.debug(f"Tool calls: {self.tool_calls}")
        
    def get_message(self) -> Dict[str, Any]:
        """Get standardized message format"""
        message = {
            "role": "assistant",
            "content": self.content
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

class LLMClient:
    """Client for interacting with OpenAI-compatible LLMs"""
    
    MAX_QUERIES_PER_CONVERSATION = 3  # GPT-4oの制限
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.tools = []
        self.messages = []
        self.system_prompt = None
        self.last_tool_calls = None
        self.query_count = 0
        
    def _prepare_messages(self) -> List[Dict[str, Any]]:
        """Prepare messages for API call"""
        formatted_messages = []
        
        if self.system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": self.system_prompt
            })
            
        formatted_messages.extend(self.messages)
        return formatted_messages
    
    async def invoke_with_prompt(self, prompt: str) -> LLMResponse:
        """Send a single prompt to the LLM"""
        if self.query_count >= self.MAX_QUERIES_PER_CONVERSATION:
            logger.warning(f"クエリ制限（{self.MAX_QUERIES_PER_CONVERSATION}回）に達しました")
            return LLMResponse(type('obj', (object,), {
                "choices": [type('obj', (object,), {
                    "finish_reason": "stop",
                    "message": type('obj', (object,), {
                        "content": f"クエリ制限（{self.MAX_QUERIES_PER_CONVERSATION}回）に達したため、これ以上の質問を処理できません。",
                        "tool_calls": None
                    })
                })]
            }))
        
        # もし前回のツール呼び出しが未処理の場合、エラーを防ぐためにダミーの応答を追加
        if self.last_tool_calls:
            for tool_call in self.last_tool_calls:
                self.messages.append({
                    "role": "tool",
                    "content": "Previous tool call was not properly handled",
                    "tool_call_id": tool_call.id
                })
            self.last_tool_calls = None
        
        self.messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = await self.invoke([])  # invokeメソッド内でクエリカウントが増加します
        if response.tool_calls:
            self.last_tool_calls = response.tool_calls
        return response
    
    async def invoke(self, tool_results: Optional[List[Dict[str, Any]]] = None) -> LLMResponse:
        """Invoke the LLM with optional tool results"""
        if tool_results:
            for result in tool_results:
                self.messages.append({
                    "role": "tool",
                    "content": str(result.get("output", "")),  # Convert to string and provide default
                    "tool_call_id": result["tool_call_id"]
                })
            # ツール結果を処理したので、last_tool_callsをクリア
            self.last_tool_calls = None
        
        if self.query_count >= self.MAX_QUERIES_PER_CONVERSATION:
            logger.warning(f"クエリ制限（{self.MAX_QUERIES_PER_CONVERSATION}回）に達しました")
            return LLMResponse(type('obj', (object,), {
                "choices": [type('obj', (object,), {
                    "finish_reason": "stop",
                    "message": type('obj', (object,), {
                        "content": f"クエリ制限（{self.MAX_QUERIES_PER_CONVERSATION}回）に達したため、これ以上の質問を処理できません。",
                        "tool_calls": None
                    })
                })]
            }))
        
        self.query_count += 1
        logger.info(f"クエリ実行回数: {self.query_count}/{self.MAX_QUERIES_PER_CONVERSATION}")
        
        try:
            try:
                logger.debug(f"送信するメッセージ: {self._prepare_messages()}")
                logger.debug(f"利用可能なツール: {self.tools}")
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=self._prepare_messages(),
                    tools=self.tools if self.tools else None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                logger.debug(f"APIレスポンス: {completion}")
            except Exception as e:
                logger.error(f"API呼び出しエラー: {str(e)}")
                raise
            logger.debug(f"使用可能なツール: {self.tools}")
            logger.debug(f"送信されたメッセージ: {self._prepare_messages()}")
        except Exception as e:
            logger.error(f"API呼び出しエラー: {str(e)}")
            raise
        
        response = LLMResponse(completion)
        self.messages.append(response.get_message())
        
        # 新しいツール呼び出しがある場合は保存
        if response.tool_calls:
            self.last_tool_calls = response.tool_calls
        
        return response
