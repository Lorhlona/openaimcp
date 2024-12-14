# src/mcp_llm_bridge/bridge.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp_llm_bridge.mcp_client import MCPClient
from mcp_llm_bridge.llm_client import LLMClient
from mcp_llm_bridge.thinking_client import ThinkingClient
import asyncio
import json
from mcp_llm_bridge.config import BridgeConfig
import logging
import colorlog
from mcp_llm_bridge.tools import DatabaseQueryTool

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
logger.setLevel(logging.DEBUG)  # DEBUGレベルに変更

class MCPLLMBridge:
    """Bridge between MCP protocol and LLM client with separate thinking process"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.mcp_client = MCPClient(config.mcp_server_params)
        self.llm_client = LLMClient(config.llm_config)
        self.thinking_client = ThinkingClient(config.get_thinking_config())
        self.query_tool = DatabaseQueryTool("test.db")
        
        # ツール実行用のプロンプト
        tool_prompt = f"""
あなたはデータベースクエリの専門家です。以下のスキーマに基づいて、
適切なSQLクエリを実行してください：

{self.query_tool.get_schema_description()}

注意点：
1. スキーマで定義された正確なカラム名を使用
2. 有効なSQLクエリを作成
3. SQLite互換の構文を使用

データベースの内容を確認する場合は、以下のようなクエリを使用してください：
- 全商品の一覧: SELECT * FROM products;
- 全カテゴリの一覧: SELECT * FROM categories;
- 特定の情報を検索: SELECT * FROM products WHERE ...;
"""
        self.llm_client.system_prompt = tool_prompt
            
        self.available_tools: List[Any] = []
        self.tool_name_mapping: Dict[str, str] = {}

    async def initialize(self):
        """Initialize both clients and set up tools"""
        try:
            await self.mcp_client.connect()
            mcp_tools = await self.mcp_client.get_available_tools()
            if hasattr(mcp_tools, 'tools'):
                self.available_tools = [*mcp_tools.tools, self.query_tool.get_tool_spec()]
            else:
                self.available_tools = [*mcp_tools, self.query_tool.get_tool_spec()]
            
            converted_tools = self._convert_mcp_tools_to_openai_format(self.available_tools)
            self.llm_client.tools = converted_tools
            
            return True
        except Exception as e:
            logger.error(f"Bridge initialization failed: {str(e)}", exc_info=True)
            return False

    def _convert_mcp_tools_to_openai_format(self, mcp_tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert MCP tool format to OpenAI tool format"""
        openai_tools = []
        
        if hasattr(mcp_tools, 'tools'):
            tools_list = mcp_tools.tools
        elif isinstance(mcp_tools, dict):
            tools_list = mcp_tools.get('tools', [])
        else:
            tools_list = mcp_tools
            
        if isinstance(tools_list, list):
            for tool in tools_list:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    openai_name = self._sanitize_tool_name(tool.name)
                    self.tool_name_mapping[openai_name] = tool.name
                    
                    tool_schema = getattr(tool, 'inputSchema', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                    
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": openai_name,
                            "description": tool.description,
                            "parameters": tool_schema
                        }
                    }
                    openai_tools.append(openai_tool)
        
        return openai_tools

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name for OpenAI compatibility"""
        return name.replace("-", "_").replace(" ", "_").lower()

    async def process_message(self, message: str) -> str:
        """Process a user message through the bridge with separate thinking process"""
        try:
            # 1. 内部分析（O1モデル）
            logger.info("=== O1モデルによる内部分析開始 ===")
            logger.info(f"入力メッセージ: {message}")
            thinking_response = await self.thinking_client.think(message)
            logger.info(f"O1モデルの分析結果: {thinking_response.content}")
            logger.info(f"ツール使用の必要性: {thinking_response.needs_tool}")
            
            # 2. ツールの使用が必要か判断
            if thinking_response.needs_tool:
                logger.info("=== GPT-4によるツール実行開始 ===")
                
                # GPT-4にツール実行を依頼
                tool_instruction = f"""
ユーザーの質問: {message}

必要な情報を取得するために、適切なデータベースクエリを実行してください。
結果は分かりやすい形式で返してください。
"""
                logger.info(f"GPT-4への指示: {tool_instruction}")
                tool_response = await self.llm_client.invoke_with_prompt(tool_instruction)
                
                tool_results = []
                # ツール実行のループ
                while tool_response.is_tool_call and tool_response.tool_calls:
                    logger.info(f"GPT-4が要求したツール実行: {tool_response.tool_calls}")
                    # ツールの実行
                    current_results = await self._handle_tool_calls(tool_response.tool_calls)
                    logger.info(f"ツール実行結果: {current_results}")
                    tool_results.extend(current_results)
                    
                    # 結果を使って次のツール呼び出しが必要か確認
                    tool_response = await self.llm_client.invoke(current_results)
                
                # 3. 最終評価（O1モデル）
                logger.info("=== O1モデルによる最終評価開始 ===")
                final_prompt = f"""
ユーザーの質問: {message}

データベースから取得した情報:
{json.dumps(tool_results, ensure_ascii=False, indent=2)}

この情報を分析し、ユーザーにとって分かりやすい形で回答を作成してください。
技術的な詳細は省略し、重要なポイントに焦点を当ててください。
"""
                logger.info(f"O1モデルへの最終プロンプト: {final_prompt}")
                final_thinking = await self.thinking_client.think(final_prompt)
                logger.info(f"O1モデルの最終回答: {final_thinking.content}")
                return final_thinking.content
                    
            else:
                # ツールが不要な場合は直接回答
                logger.info("ツール使用不要と判断、直接回答を返します")
                return thinking_response.content
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"申し訳ありません。処理中にエラーが発生しました: {str(e)}"

    async def _handle_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle tool calls through MCP"""
        tool_responses = []
        
        for tool_call in tool_calls:
            try:
                openai_name = tool_call.function.name
                mcp_name = self.tool_name_mapping.get(openai_name)
                
                if not mcp_name:
                    raise ValueError(f"Unknown tool: {openai_name}")
                
                arguments = json.loads(tool_call.function.arguments)
                logger.info(f"ツール実行: {mcp_name}, 引数: {arguments}")
                result = await self.mcp_client.call_tool(mcp_name, arguments)
                
                if isinstance(result, str):
                    output = result
                elif hasattr(result, 'content') and isinstance(result.content, list):
                    output = " ".join(
                        content.text for content in result.content 
                        if hasattr(content, 'text')
                    )
                else:
                    output = str(result)
                
                logger.info(f"ツール実行結果: {output}")
                
                tool_responses.append({
                    "tool_call_id": tool_call.id,
                    "output": output
                })
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                tool_responses.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Error: {str(e)}"
                })
        
        return tool_responses

    async def close(self):
        """Clean up resources"""
        await self.mcp_client.__aexit__(None, None, None)

class BridgeManager:
    """Manager class for handling the bridge lifecycle"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.bridge: Optional[MCPLLMBridge] = None

    async def __aenter__(self) -> MCPLLMBridge:
        """Context manager entry"""
        self.bridge = MCPLLMBridge(self.config)
        await self.bridge.initialize()
        return self.bridge
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.bridge:
            await self.bridge.close()
