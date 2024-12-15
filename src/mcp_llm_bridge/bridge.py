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
from mcp_llm_bridge.tools import DatabaseQueryTool, GoogleSearchTool, HumanTool

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
        self.search_tool = GoogleSearchTool()
        self.human_tool = HumanTool()
        
        # ツール実行用のプロンプトを生成
        self._create_tool_prompt()
            
        self.available_tools: List[Any] = []
        self.tool_name_mapping: Dict[str, str] = {}

    def _create_tool_prompt(self):
        """ツール実行用のプロンプトを生成"""
        tool_prompt = f"""
あなたは高度なアシスタントで、データベースクエリ、Google検索、およびユーザーとのインタラクションの専門家です。
与えられた指示に従って、利用可能なファンクションを使用してタスクを実行してください。

【利用可能なファンクション】
1. database_query
- 説明: SQLiteデータベースに対してクエリを実行
- パラメータ: query (string) - 実行するSQLクエリ
- 戻り値: クエリ結果のJSON形式データ
- 使用例: {{"query": "SELECT * FROM products LIMIT 1;"}}

2. google_search
- 説明: Google検索を実行して関連する結果を取得
- パラメータ:
  - query (string) - 検索クエリ文字列（商品名、モデル、価格などの具体的な情報を含める）
  - num_results (integer, optional) - 取得する結果の数（1-10、デフォルト5）
- 戻り値: 検索結果のJSON形式データ
- 使用例:
  - 基本検索: {{"query": "Laptop Pro X price specs reviews 2024", "num_results": 5}}
  - 価格検索: {{"query": "Laptop Pro X current price sale", "num_results": 3}}

3. human_interaction
- 説明: ユーザーに追加の質問をして情報を収集
- パラメータ: question (string) - ユーザーへの質問文
- 戻り値: ユーザーの回答を含むJSON形式データ
- 使用例: {{"question": "どの写真集をお探しですか？"}}
- 使用ガイドライン:
  - 質問は自然な会話形式で
  - 一度に1つの情報のみを確認
  - 情報が不足している場合は最優先で使用

【データベーススキーマ】
{self.query_tool.get_schema_description()}

【実行ルール】
1. 1フェーズで最大3つまでのクエリまたは検索を実行
2. データベースクエリの場合:
   - スキーマで定義された正確なカラム名を使用
   - SQLite互換の構文を使用
3. Google検索の場合:
   - 具体的で詳細な検索クエリを使用（商品名、モデル、年など）
   - 価格情報を取得する場合は "price"、"cost"、"sale" などのキーワードを含める
   - 必要に応じて結果数を指定
4. human_interactionの場合:
   - 他のツールと組み合わせず、単独で使用
   - ユーザーからの回答を待ってから次の処理を決定
5. 未定義のファンクションは使用不可

【実行例】
1. データベース操作:
  database_query({{"query": "SELECT * FROM products WHERE category = 'Electronics';"}})

2. Google検索:
  google_search({{"query": "Laptop Pro X latest price 2024 review", "num_results": 5}})

3. ユーザーとの対話:
  human_interaction({{"question": "どの商品についての情報をお探しですか？"}})

実行フェーズで指定されたクエリのみを実行し、3クエリの制限を厳守してください。
"""
        self.llm_client.system_prompt = tool_prompt

    async def initialize(self):
        """Initialize both clients and set up tools"""
        try:
            await self.mcp_client.connect()
            mcp_tools = await self.mcp_client.get_available_tools()
            
            try:
                # ツールの仕様を取得
                query_tool_spec = self.query_tool.get_tool_spec()
                search_tool_spec = self.search_tool.get_tool_spec()
                human_tool_spec = self.human_tool.get_tool_spec()
                
                logger.debug(f"データベースツール仕様: {query_tool_spec}")
                logger.debug(f"検索ツール仕様: {search_tool_spec}")
                logger.debug(f"対話ツール仕様: {human_tool_spec}")
                
                # ツール名のマッピングを設定
                self.tool_name_mapping = {
                    "database_query": query_tool_spec["name"],
                    "google_search": search_tool_spec["name"],
                    "human_interaction": human_tool_spec["name"]
                }
                
                # 利用可能なツールを設定
                self.available_tools = [query_tool_spec, search_tool_spec, human_tool_spec]
                
                logger.debug(f"ツール名マッピング: {self.tool_name_mapping}")
                logger.debug(f"利用可能なツール: {self.available_tools}")
            except Exception as e:
                logger.error(f"ツール初期化エラー: {str(e)}")
                raise
            
            # OpenAI形式に変換
            converted_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "human_interaction",
                        "description": "ユーザーに追加の質問をして情報を収集するツール",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "ユーザーへの質問"
                                }
                            },
                            "required": ["question"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "google_search",
                        "description": "Google検索を実行して関連する結果を取得",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "検索クエリ文字列"
                                },
                                "num_results": {
                                    "type": "integer",
                                    "description": "取得する結果の数（1-10）",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "database_query",
                        "description": "データベースに対してクエリを実行",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "実行するSQLクエリ"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
            logger.debug(f"利用可能なツール: {converted_tools}")
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
            max_iterations = 2  # 最大2回の追加実行が可能（初回 + 最大2回）
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
                        logger.warning("実行フェーズ情報が見つかりません。デフォルトの実行を使用します。")
                        phase_info = (
                            "現在のフェーズ: 1/1\n"
                            "実行すべき操作:\n"
                            "1. 基本的な情報収集\n"
                            "2. 詳細データの取得\n"
                            "3. 結果の確認と検証"
                        )

                    current_context = (
                        f"EXECUTION_PHASE:\n"
                        f"{phase_info}\n\n"
                        f"CONTEXT:\n"
                        f"- User Query: {original_message}\n"
                        f"- Previous Results: {json.dumps(accumulated_results, ensure_ascii=False, indent=2) if accumulated_results else 'None'}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"1. Execute the specified operations in order\n"
                        f"2. Do not exceed 3 operations per phase\n"
                        f"3. Use only the available functions (database_query, google_search, or human_interaction)\n"
                        f"4. When using human_interaction, wait for user response before proceeding\n"
                        f"5. Return results in JSON format"
                    )
                    logger.info(f"GPT-4への指示: {current_context}")
                    tool_response = await self.llm_client.invoke_with_prompt(current_context)

                    # ツール実行のループ
                    while tool_response.is_tool_call and tool_response.tool_calls:
                        logger.info(f"GPT-4が要求したツール実行: {tool_response.tool_calls}")
                        
                        # human_interactionが必要な場合は強制的にそれを使用
                        if "[Human_Interaction]" in phase_info:
                            question = phase_info.split("[Human_Interaction]")[1].split("\n")[0].strip()
                            if not question.endswith("？"):
                                question += "？"
                            human_call = {
                                "id": "forced_human_interaction",
                                "name": "human_interaction",
                                "arguments": json.dumps({
                                    "question": question
                                })
                            }
                            current_results = await self._handle_tool_calls([human_call])
                            logger.info(f"ユーザーとの対話結果: {current_results}")
                            accumulated_results.extend(current_results)
                            
                            # ユーザーの回答を元に新しい思考プロセスを開始
                            user_response = json.loads(current_results[0]["output"])
                            message = f"""
元の質問: {original_message}

ユーザーからの回答:
{json.dumps(user_response, ensure_ascii=False, indent=2)}

これまでの実行結果:
{json.dumps(accumulated_results, ensure_ascii=False, indent=2)}

この情報を元に、次に必要な操作を判断してください。
"""
                            break
                        else:
                            # GPT-4が選択したツールがhuman_interactionの場合
                            if any(call.function.name == "human_interaction" for call in tool_response.tool_calls):
                                human_call = next(call for call in tool_response.tool_calls if call.function.name == "human_interaction")
                                current_results = await self._handle_tool_calls([{
                                    "id": human_call.id,
                                    "name": human_call.function.name,
                                    "arguments": human_call.function.arguments
                                }])
                                logger.info(f"ユーザーとの対話結果: {current_results}")
                                accumulated_results.extend(current_results)
                                
                                # ユーザーの回答を元に新しい思考プロセスを開始
                                user_response = json.loads(current_results[0]["output"])
                                message = f"""
元の質問: {original_message}

ユーザーからの回答:
{json.dumps(user_response, ensure_ascii=False, indent=2)}

これまでの実行結果:
{json.dumps(accumulated_results, ensure_ascii=False, indent=2)}

この情報を元に、次に必要な操作を判断してください。
"""
                                break
                            else:
                                # 通常のツール実行
                                tool_calls_dict = [{
                                    "id": call.id,
                                    "name": call.function.name,
                                    "arguments": call.function.arguments
                                } for call in tool_response.tool_calls]
                                current_results = await self._handle_tool_calls(tool_calls_dict)
                                logger.info(f"ツール実行結果: {current_results}")
                                accumulated_results.extend(current_results)

                                # 結果を使って次のツール呼び出しが必要か確認
                                tool_response = await self.llm_client.invoke(current_results)

                    # 次のイテレーションのためにメッセージを更新
                    if not message:  # human_interactionで更新されていない場合
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
                formatted_response += "操作が完了しました。\n\n"
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
                openai_name = tool_call["name"]
                mcp_name = self.tool_name_mapping.get(openai_name)
                
                if not mcp_name:
                    raise ValueError(f"Unknown tool: {openai_name}")
                
                arguments = json.loads(tool_call["arguments"])
                logger.info(f"ツール実行: {mcp_name}, 引数: {arguments}")
                
                # ツールの種類に基づいて実行
                logger.debug(f"ツール実行要求: {openai_name}, 引数: {arguments}")
                try:
                    if openai_name == "human_interaction":
                        result = await self.human_tool.execute(arguments)
                        logger.debug(f"human_interaction実行結果: {result}")
                    elif openai_name == "google_search":
                        result = await self.search_tool.execute(arguments)
                        logger.debug(f"google_search実行結果: {result}")
                    elif openai_name == "database_query":
                        result = await self.query_tool.execute(arguments)
                        logger.debug(f"database_query実行結果: {result}")
                    else:
                        logger.warning(f"未知のツール名: {openai_name}")
                        result = f"Error: Unknown tool {openai_name}"
                except Exception as e:
                    logger.error(f"ツール実行エラー: {str(e)}")
                    result = f"Error: {str(e)}"
                
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
                    "tool_call_id": tool_call["id"],
                    "output": output
                })
                
            except Exception as e:
                logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
                tool_responses.append({
                    "tool_call_id": tool_call["id"],
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
