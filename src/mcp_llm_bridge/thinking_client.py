from typing import Dict, List, Any, Optional
import json
import openai
from mcp_llm_bridge.config import LLMConfig
from mcp_llm_bridge.schemas import ThinkingResponse, TaskPlan, TaskPhase, Operation
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
logger.setLevel(logging.DEBUG)

class ThinkingClient:
    """O1モデル用の思考プロセス専用クライアント"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.task_plan: Optional[TaskPlan] = None
        self._context = """
あなたは高度な思考エンジンとして、ユーザーの要求を分析し、実行計画を立案します。
応答は必ず以下のJSON形式で返してください：

{
    "task_plan": {  // 初回のみ必須、2回目以降は省略可
        "overall_tasks": [  // 達成すべきサブタスクのリスト
            "ユーザーの求める情報の特定",
            "必要なデータの収集",
            "結果の集約と回答生成"
        ],
        "total_phases": 3,  // 予定される総フェーズ数
        "phases": [  // 各フェーズの詳細
            {
                "phase_number": 1,
                "operations": [
                    {
                        "type": "human_interaction",
                        "parameters": {
                            "question": "具体的にどのような情報をお探しですか？"
                        }
                    }
                ],
                "description": "ユーザーの意図を明確化"
            }
        ]
    },
    "current_phase": {  // 現在実行すべきフェーズの情報
        "phase_number": 1,
        "operations": [
            {
                "type": "human_interaction",
                "parameters": {
                    "question": "具体的にどのような情報をお探しですか？"
                }
            }
        ],
        "description": "ユーザーの意図を明確化"
    },
    "needs_tool": true,  // ツール使用の必要性
    "task_completed": false,  // タスク完了状態
    "final_response": null  // タスク完了時のみ設定
}

# 利用可能なツール
1. human_interaction
   - parameters: {"question": "質問文"}
   - 制約: 1フェーズで1つの質問のみ

2. database_query
   - parameters: {"query": "SQLクエリ"}
   - テーブル: products, categories
   - 制約: 適切なSQLite構文

3. google_search
   - parameters: {"query": "検索文", "num_results": 件数}
   - 制約: num_resultsは1-10の範囲

# 実行ルール
1. 1フェーズで最大3つまでの操作
2. human_interactionは単独で使用
3. database_queryとgoogle_searchは組み合わせ可能
4. 最大5フェーズまで
5. 各フェーズは明確な目的が必要

# エラー処理
- 不正なJSON形式の場合は再プロンプト
- 未定義のツール使用は禁止
- パラメータの検証は必須
"""
        logger.info(f"ThinkingClient初期化完了: モデル={config.model}")
    
    async def think(self, context: str, tool_result: Optional[str] = None, iteration: int = 0) -> ThinkingResponse:
        """思考プロセスの実行"""
        logger.info(f"=== O1モデルの思考プロセス開始 (イテレーション: {iteration}) ===")
        
        if tool_result:
            prompt = f"""
{self._context}

ユーザーからの質問:
{context}

前回の実行結果:
{tool_result}

これはイテレーション{iteration}回目です。
前回の実行結果を分析し、次のフェーズの実行計画または最終応答を決定してください。
タスクが完了した場合は、final_responseに結果をまとめてください。
"""
        else:
            prompt = f"""
{self._context}

ユーザーからの質問:
{context}

これは最初の分析です。
1. まずタスク全体を分解し、必要なサブタスクを列挙
2. 各フェーズで実行する操作を計画
3. 最初のフェーズの実行計画を決定

task_planとcurrent_phaseの両方を含むJSONで応答してください。
"""
        
        try:
            logger.info("O1 APIリクエスト開始")
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_completion_tokens=32768
            )
            logger.info("O1 APIリクエスト完了")
            
            # レスポンスの解析と構造化
            response_content = completion.choices[0].message.content
            logger.debug(f"生の応答内容: {response_content}")

            try:
                # JSON形式の応答を探す
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_content = response_content[json_start:json_end]
                    response_dict = json.loads(json_content)
                else:
                    # JSON形式でない場合は、構造化された応答を生成
                    response_dict = {
                        "current_phase": {
                            "phase_number": 1,
                            "operations": [
                                {
                                    "type": "human_interaction",
                                    "parameters": {
                                        "question": "具体的にどのような情報をお探しですか？"
                                    }
                                }
                            ],
                            "description": "ユーザーの意図を明確化"
                        },
                        "needs_tool": True,
                        "task_completed": False,
                        "final_response": None
                    }
                    if iteration == 0:
                        response_dict["task_plan"] = {
                            "overall_tasks": ["ユーザーの意図を明確化", "必要な情報の収集", "結果の整理と提示"],
                            "total_phases": 3,
                            "phases": [response_dict["current_phase"]]
                        }

                # Pydanticモデルに変換
                thinking_response = ThinkingResponse(**response_dict)
                
                # 初回の場合はタスク計画を保存
                if iteration == 0 and thinking_response.task_plan:
                    self.task_plan = thinking_response.task_plan
                
                logger.info(f"思考プロセスの結果: ツール使用必要={thinking_response.needs_tool}, タスク完了={thinking_response.task_completed}")
                return thinking_response
                
            except Exception as e:
                logger.error(f"応答の解析でエラー: {str(e)}")
                # エラー時はデフォルトの応答を返す
                default_response = ThinkingResponse(**{
                    "current_phase": {
                        "phase_number": 1,
                        "operations": [
                            {
                                "type": "human_interaction",
                                "parameters": {
                                    "question": "申し訳ありません。もう一度具体的に教えていただけますか？"
                                }
                            }
                        ],
                        "description": "ユーザーの意図を再確認"
                    },
                    "needs_tool": True,
                    "task_completed": False,
                    "final_response": None
                })
                return default_response
                
        except Exception as e:
            logger.error(f"思考プロセスでエラー発生: {str(e)}")
            raise
