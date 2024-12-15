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
あなたはデータベースクエリの専門家です。与えられた指示に従って、
利用可能なファンクションを使用してクエリを実行してください。

【利用可能なファンクション】
1. database_query
- 説明: SQLiteデータベースに対してクエリを実行
- パラメータ: query (string) - 実行するSQLクエリ
- 戻り値: クエリ結果のJSON形式データ

【データベーススキーマ】
{self.query_tool.get_schema_description()}

【実行ルール】
1. 1フェーズで最大3つまでのクエリ実行
2. スキーマで定義された正確なカラム名を使用
3. SQLite互換の構文を使用
4. 未定義のファンクションは使用不可

【クエリ例】
- 全商品の一覧: SELECT * FROM products;
- 全カテゴリの一覧: SELECT * FROM categories;
- 特定の情報を検索: SELECT * FROM products WHERE ...;

実行フェーズで指定されたクエリのみを実行し、3クエリの制限を厳守してください。
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
            iteration = 0
            max_iterations = 3  # 最大3回まで実行可能（初回 + 最大2回の追加実行）
            accumulated_results = []
            original_message = message
            final_response = None

            while iteration < max_iterations:
                logger.info(f"=== 実行イテレーション {iteration + 1}/{max_iterations} ===")

                # 1. 内部分析（O1モデル）
                logger.info("=== O1モデルによる内部分析開始 ===")
                thinking_response = await self.thinking_client.think(
                    message,
                    tool_result=json.dumps(accumulated_results, ensure_ascii=False, indent=2) if accumulated_results else None,
                    iteration=iteration
                )
                logger.info(f"O1モデルの分析結果: {thinking_response.content}")
                logger.info(f"ツール使用の必要性: {thinking_response.needs_tool}")
                logger.info(f"タスク完了状態: {thinking_response.task_completed}")

                # O1の応答タイプを取得
                response_type = thinking_response._get_response_type()
                final_response = thinking_response.content

                # 応答タイプに基づいて処理を分岐
                if response_type in ["会話応答", "ガイダンス"]:
                    logger.info(f"{response_type}として処理します")
                    return final_response

                # タスクが完了している場合は終了
                if thinking_response.task_completed:
                    logger.info("タスク完了、処理を終了します")
                    return self._format_final_response(final_response, accumulated_results)

                # ツール操作が必要な場合
                if thinking_response.needs_tool:
                    logger.info("=== GPT-4によるツール実行開始 ===")

                    # GPT-4にツール実行を依頼
                    # O1の分析から実行フェーズ情報を抽出
                    try:
                        phase_info = thinking_response.content.split("【実行フェーズ】")[1].split("【詳細情報】")[0].strip()
                    except IndexError:
                        # フェーズ情報が見つからない場合のフォールバック
                        logger.warning("実行フェーズ情報が見つかりません。デフォルトのクエリを使用します。")
                        phase_info = (
                            "現在のフェーズ: 1/1\n"
                            "実行すべきクエリ:\n"
                            "1. テーブル構造の確認 (SELECT * FROM sqlite_master WHERE type='table')\n"
                            "2. 商品データの取得 (SELECT * FROM products)\n"
                            "3. カテゴリ情報の取得 (SELECT * FROM categories)"
                        )

                    current_context = (
                        f"EXECUTION_PHASE:\n"
                        f"{phase_info}\n\n"
                        f"CONTEXT:\n"
                        f"- User Query: {original_message}\n"
                        f"- Previous Results: {json.dumps(accumulated_results, ensure_ascii=False, indent=2) if accumulated_results else 'None'}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1. Execute the specified queries in order\n"
                        f"2. Do not exceed 3 queries per phase\n"
                        f"3. Use only the available database_query function\n"
                        f"4. Return results in JSON format"
                    )
                    logger.info(f"GPT-4への指示: {current_context}")
                    tool_response = await self.llm_client.invoke_with_prompt(current_context)

                    # ツール実行のループ
                    while tool_response.is_tool_call and tool_response.tool_calls:
                        logger.info(f"GPT-4が要求したツール実行: {tool_response.tool_calls}")
                        current_results = await self._handle_tool_calls(tool_response.tool_calls)
                        logger.info(f"ツール実行結果: {current_results}")
                        accumulated_results.extend(current_results)

                        # 結果を使って次のツール呼び出しが必要か確認
                        tool_response = await self.llm_client.invoke(current_results)

                    # 次のイテレーションのためにメッセージを更新
                    message = f"""
元の質問: {original_message}

これまでの実行結果:
{json.dumps(accumulated_results, ensure_ascii=False, indent=2)}

この情報を元に、タスクが完了したか、さらなる情報が必要か判断してください。
"""
                else:
                    # ツールが不要な場合は直接回答
                    logger.info("ツール使用不要と判断、直接回答を返します")
                    return self._format_final_response(final_response, accumulated_results)

                iteration += 1

            # 最大イテレーション回数に達した場合の最終評価
            logger.info("=== 最大イテレーション回数到達、最終評価実行 ===")
            final_prompt = f"""
ユーザーの質問: {original_message}

収集した全ての情報:
{json.dumps(accumulated_results, ensure_ascii=False, indent=2)}

最大実行回数（{max_iterations}回）に達しました。
現在の情報を元に、最終的な回答を作成してください。
"""
            final_thinking = await self.thinking_client.think(final_prompt, iteration=max_iterations)
            return self._format_final_response(final_thinking.content, accumulated_results)

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"申し訳ありません。処理中にエラーが発生しました: {str(e)}"

    def _format_final_response(self, response: str, results: List[Dict[str, Any]]) -> str:
        """Format the final response with execution results"""
        if not response or not "【実行結果】" in response:
            # レスポンスが標準フォーマットでない場合、整形する
            formatted_response = "【実行結果】\n"
            if results:
                formatted_response += "データベースの操作が完了しました。\n\n"
                formatted_response += "【タスク状態】\n"
                formatted_response += "タスクは完了しました。\n\n"
                formatted_response += "【詳細情報】\n"
                try:
                    # 結果の整形
                    for result in results:
                        if "output" in result:
                            try:
                                # JSON文字列をパース
                                output_data = json.loads(result["output"].replace("'", '"'))
                                formatted_response += json.dumps(output_data, ensure_ascii=False, indent=2)
                            except:
                                # パースに失敗した場合は生の出力を使用
                                formatted_response += result["output"]
                            formatted_response += "\n"
                except Exception as e:
                    logger.error(f"結果の整形中にエラーが発生: {str(e)}")
                    formatted_response += str(results)
            else:
                formatted_response += response

            return formatted_response
        return response

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
