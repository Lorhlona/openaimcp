from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp_llm_bridge.mcp_client import MCPClient
from mcp_llm_bridge.llm_client import LLMClient
from mcp_llm_bridge.thinking_client import ThinkingClient
from mcp_llm_bridge.schemas import ThinkingResponse, TaskPlan, TaskPhase, Operation, ExecutionResult
import asyncio
import json
from mcp_llm_bridge.config import BridgeConfig
import logging
import colorlog
from mcp_llm_bridge.tools import DatabaseQueryTool, GoogleSearchTool, HumanTool
from mcp_llm_bridge.tools.spotify import SpotifyTool
from mcp_llm_bridge.voice_manager import VoiceManager

def setup_logging():
    """ロギングの設定を一度だけ行う"""
    root_logger = logging.getLogger()
    
    # ハンドラーが既に設定されている場合はスキップ
    if root_logger.handlers:
        return
        
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
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

# ロギングの設定を行う
setup_logging()

# モジュールのロガーを取得
logger = logging.getLogger(__name__)

class MCPLLMBridge:
    """Bridge between MCP protocol and LLM client with structured thinking process"""
    
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.mcp_client = MCPClient(config.mcp_server_params)
        self.llm_client = LLMClient(config.llm_config)
        self.thinking_client = ThinkingClient(config.get_thinking_config())
        self.query_tool = DatabaseQueryTool("test.db")
        self.search_tool = GoogleSearchTool()
        self.human_tool = HumanTool()
        self.spotify_tool = SpotifyTool()
        
        # 音声マネージャーの初期化
        try:
            self.voice_manager = VoiceManager()
            logger.info("音声マネージャーの初期化成功")
        except Exception as e:
            self.voice_manager = None
            logger.error(f"音声マネージャーの初期化失敗: {str(e)}")
        
        # ツール実行用のプロンプトを生成
        self._create_tool_prompt()
        
        self.available_tools: List[Any] = []
        self.tool_name_mapping: Dict[str, str] = {}
        self.current_task_plan: Optional[TaskPlan] = None
        self.spotify_state = {
            "is_playing": False,
            "current_track_id": None,
            "current_device": None
        }
        self.is_task_completed = False    # タスクが完了したかどうかを表すフラグ

    def _create_tool_prompt(self):
        """ツール実行用のプロンプトを生成"""
        tool_prompt = f"""
あなたは高度なアシスタントで、与えられた実行計画に従ってツールを実行する専門家です。
各操作は正確に実行し、結果を適切にフォーマットしてください。

【利用可能なツール】
1. database_query
- 説明: SQLiteデータベースに対してクエリを実行
- パラメータ: query (string) - 実行するSQLクエリ
- 戻り値: クエリ結果のJSON形式データ

2. google_search
- 説明: Google検索を実行して関連する結果を取得
- パラメータ:
  - query (string) - 検索クエリ文字列
  - num_results (integer, optional) - 取得する結果の数（1-10）
- 戻り値: 検索結果のJSON形式データ

3. human_interaction
- 説明: ユーザーに追加の質問をして情報を収集
- パラメータ: question (string) - ユーザーへの質問文
- 戻り値: ユーザーの回答を含むJSON形式データ

4. spotify
- 説明: Spotifyの操作を行う
- パラメータ:
  - action (string) - 実行するアクション（search/play/pause/current_track/add_to_queue）
  - query (string, optional) - 検索クエリ（searchアクション用）
  - track_id (string, optional) - トラックID（play/add_to_queueアクション用）
- 戻り値: アクションの結果をJSON形式で返す

【データベーススキーマ】
{self.query_tool.get_schema_description()}

【実行ルール】
1. 与えられた操作のみを実行
2. パラメータは厳密に検証
3. エラー時は適切なエラーメッセージを返す
4. 結果は常にJSON形式で返す

実行計画に従って各操作を実行し、結果を収集してください。
"""
        self.llm_client.system_prompt = tool_prompt

    async def initialize(self):
        """Initialize both clients and set up tools"""
        try:
            await self.mcp_client.connect()
            
            # ツールの仕様を取得
            query_tool_spec = self.query_tool.get_tool_spec()
            search_tool_spec = self.search_tool.get_tool_spec()
            human_tool_spec = self.human_tool.get_tool_spec()
            spotify_tool_spec = self.spotify_tool.get_tool_spec()
            
            # ツール名のマッピングを設定
            self.tool_name_mapping = {
                "database_query": query_tool_spec["name"],
                "google_search": search_tool_spec["name"],
                "human_interaction": human_tool_spec["name"],
                "spotify": spotify_tool_spec["name"]
            }
            
            # 利用可能なツールを設定
            self.available_tools = [query_tool_spec, search_tool_spec, human_tool_spec, spotify_tool_spec]
            
            # OpenAI形式のツール定義
            converted_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "human_interaction",
                        "description": "ユーザーに追加の質問をして情報を収集",
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
                },
                {
                    "type": "function",
                    "function": {
                        "name": "spotify",
                        "description": "Spotifyの操作を行う",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["search", "play", "pause", "current_track", "add_to_queue"],
                                    "description": "実行するアクション"
                                },
                                "query": {
                                    "type": "string",
                                    "description": "検索クエリ（searchアクション用）"
                                },
                                "track_id": {
                                    "type": "string",
                                    "description": "トラックID（play/add_to_queueアクション用）"
                                }
                            },
                            "required": ["action"]
                        }
                    }
                }
            ]
            
            self.llm_client.tools = converted_tools
            return True
            
        except Exception as e:
            logger.error(f"Bridge initialization failed: {str(e)}", exc_info=True)
            return False

    async def process_message(self, user_input: str) -> str:
        """Process a user message through the bridge with structured thinking process"""
        try:
            # ユーザー発話をThinkingClientに記録
            self.thinking_client.add_user_message(user_input)

            iteration = 0
            max_iterations = 4
            accumulated_results: List[Dict[str, Any]] = []

            # 最初の思考プロセス (iteration=0)
            thinking_response = await self.thinking_client.think(user_input)

            while iteration < max_iterations:
                logger.info(f"=== 実行イテレーション {iteration + 1}/{max_iterations} ===")

                # タスク完了チェック
                if thinking_response.task_completed:
                    self.is_task_completed = True
                    final_response = thinking_response.final_response
                    if not final_response:
                        final_response = self._format_final_response(accumulated_results)
                    if not final_response:
                        final_response = "申し訳ありません。応答を生成できませんでした。"

                    # 音声出力
                    if final_response and self.voice_manager and self.voice_manager.is_voice_enabled():
                        try:
                            logger.debug("音声出力を開始します")
                            self.voice_manager.process_text(final_response)
                            logger.debug("音声出力が完了しました")
                        except Exception as e:
                            logger.error(f"音声出力でエラー: {str(e)}")

                    return final_response

                # ツール実行が必要な場合
                if thinking_response.needs_tool and thinking_response.current_phase:
                    current_results = await self._execute_phase(thinking_response.current_phase)
                    # accumulated_resultsに追加しておく
                    for result in current_results:
                        result_dict = {
                            "operation_type": result.operation_type,
                            "success": result.success,
                            "result": result.result,
                            "error": result.error
                        }
                        accumulated_results.append(result_dict)

                    # --- ツール結果の日本語要約をthinking_clientに追加 ---
                    if current_results:
                        results_count = len(current_results)
                        jp_tool_summary = f"前回のイテレーションでは、合計 {results_count} 件の操作を行いました。"
                        for i, r in enumerate(current_results, start=1):
                            if r.success:
                                jp_tool_summary += f"\n・{i}番目: '{r.operation_type}' を実行して成功しました。"
                            else:
                                jp_tool_summary += f"\n・{i}番目: '{r.operation_type}' を実行しましたが失敗しました (エラー: {r.error})."
                        self.thinking_client.add_assistant_message(f"【要約】{jp_tool_summary}")
                    # -----------------------------------------------

                    # ツール実行後に改めて思考プロセス
                    thinking_response = await self.thinking_client.think(
                        user_input,
                        json.dumps(accumulated_results, ensure_ascii=False, indent=2),
                        iteration + 1
                    )
                else:
                    # ツールが不要な場合は直接応答
                    final_response = thinking_response.final_response
                    if not final_response:
                        final_response = "申し訳ありません。応答を生成できませんでした。"

                    # 音声出力
                    if final_response and self.voice_manager and self.voice_manager.is_voice_enabled():
                        try:
                            logger.debug("音声出力を開始します")
                            self.voice_manager.process_text(final_response)
                            logger.debug("音声出力が完了しました")
                        except Exception as e:
                            logger.error(f"音声出力でエラー: {str(e)}")

                    return final_response

                iteration += 1

            # max_iterationsを超えた場合
            if self.is_task_completed:
                return final_response or "タスクが完了しました。"
            else:
                return "タスクを完了できませんでした。"

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return f"申し訳ありません。処理中にエラーが発生しました: {str(e)}"

    async def _execute_phase(self, phase: TaskPhase) -> List[ExecutionResult]:
        """Execute a single phase of operations"""
        results = []
        
        for operation in phase.operations:
            try:
                if operation.type == "spotify":
                    action = operation.parameters.get("action")
                    
                    # 現在の状態をチェック
                    if action == "play" and self.spotify_state["is_playing"]:
                        # 既に再生中の場合はスキップ
                        continue
                    elif action == "pause":
                        # 一時停止時は状態をリセット
                        self.spotify_state = {
                            "is_playing": False,
                            "current_track_id": None,
                            "current_device": None
                        }
                    
                    # Spotifyツールを実行
                    result = await self.spotify_tool.execute(operation.parameters)
                    
                    # 結果に基づいて状態を更新
                    if result.get("status") == "playing":
                        self.spotify_state = {
                            "is_playing": True,
                            "current_track_id": result.get("track_id"),
                            "current_device": result.get("device")
                        }
                    
                elif operation.type == "human_interaction":
                    result = await self.human_tool.execute(operation.parameters)
                elif operation.type == "google_search":
                    result = await self.search_tool.execute(operation.parameters)
                elif operation.type == "database_query":
                    result = await self.query_tool.execute(operation.parameters)
                else:
                    raise ValueError(f"Unknown operation type: {operation.type}")
                
                execution_result = ExecutionResult(
                    operation_type=operation.type,
                    success=True,
                    result=result,
                    error=None
                )
                
            except Exception as e:
                logger.error(f"Operation execution failed: {str(e)}")
                execution_result = ExecutionResult(
                    operation_type=operation.type,
                    success=False,
                    result=None,
                    error=str(e)
                )
            
            results.append(execution_result)
            
            # human_interactionの場合は即座に返す
            if operation.type == "human_interaction":
                break
        
        return results

    def _format_final_response(self, results: List[Dict[str, Any]]) -> str:
        """Format the final response with execution results"""
        # 最後の結果を確認
        if results and results[-1]["success"] and results[-1]["result"]:
            last_result = results[-1]
            
            # Google検索結果の場合
            if last_result["operation_type"] == "google_search":
                try:
                    search_results = last_result["result"]
                    if isinstance(search_results, str):
                        search_results = json.loads(search_results)
                    
                    response = "検索結果:\n\n"
                    for item in search_results:
                        if isinstance(item, dict):
                            title = item.get('title', '')
                            snippet = item.get('snippet', '')
                            if title and snippet:
                                response += f"・{title}\n{snippet}\n\n"
                    
                    return response.strip()
                except:
                    pass
            
            # その他の結果の場合はデフォルトフォーマットを使用
            try:
                if isinstance(last_result["result"], str):
                    return json.loads(last_result["result"])
                return str(last_result["result"])
            except:
                return str(last_result["result"])
        
        # エラーまたは結果がない場合
        return "申し訳ありません。結果を取得できませんでした。"

    async def close(self):
        """Clean up resources"""
        await self.mcp_client.__aexit__(None, None, None)

    def summarize_context(self) -> str:
        """
        これまでの会話・ツール結果などを要約して返す。
        ThinkingClientの会話履歴と実行結果を利用。
        """
        return self.thinking_client.get_conversation_summary()

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

    @property
    def is_task_completed(self) -> bool:
        """bridgeがタスク完了しているかどうかを外部から確認できるように。"""
        if not self.bridge:
            return False
        return self.bridge.is_task_completed

    def summarize_context(self) -> str:
        """bridgeのsummarize_contextを呼び出し、要約を取得。"""
        if not self.bridge:
            return ""
        return self.bridge.summarize_context()
