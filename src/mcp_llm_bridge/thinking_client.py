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
logger.setLevel(logging.DEBUG)  # DEBUGレベルに変更

class ThinkingResponse:
    """思考プロセスの応答フォーマット"""
    def __init__(self, completion: Any):
        self.completion = completion
        self.content = completion.choices[0].message.content
        self.task_completed = self._analyze_task_completion()
        self.needs_tool = self._analyze_needs_tool() if not self.task_completed else False
        logger.debug(f"生のレスポンス: {completion}")
        logger.debug(f"処理済みコンテンツ: {self.content}")
        
    def _analyze_needs_tool(self) -> bool:
        """思考内容からツール使用の必要性を判断"""
        # タスクが完了している場合は常にFalse
        if self.task_completed:
            return False
            
        # 追加の情報収集が必要なことを示す表現
        tool_indicators = [
            "データベースの検索が必要",
            "情報を取得する必要があります",
            "確認が必要です",
            "調べる必要があります",
            "追加の情報が必要です",
            "データベースを確認する必要があります"
        ]
        needs_tool = any(indicator in self.content.lower() for indicator in tool_indicators)
        logger.debug(f"ツール使用の必要性判断: {needs_tool}")
        return needs_tool

    def _analyze_task_completion(self) -> bool:
        """タスクが完了したかどうかを判断"""
        # 完了を示す表現
        completion_indicators = [
            "タスクは完了しました",
            "タスクは正常に完了しました",
            "これで完了しました",
            "追加が正常に完了しました",
            "確認が完了しました"
        ]
        
        # 未完了を示す表現
        incomplete_indicators = [
            "追加の情報が必要です",
            "さらなる確認が必要です",
            "まだ完了していません",
            "もう少し調査が必要です",
            "情報が不十分です",
            "確認できていません"
        ]
        
        is_complete = any(indicator in self.content.lower() for indicator in completion_indicators)
        is_incomplete = any(indicator in self.content.lower() for indicator in incomplete_indicators)
        
        # 完了の表現があり、かつ未完了の表現がない場合にのみ完了とみなす
        task_completed = is_complete and not is_incomplete
        logger.debug(f"タスク完了判断: {task_completed}")
        return task_completed

    def get_message(self) -> Dict[str, Any]:
        """標準化されたメッセージフォーマット"""
        return {
            "role": "assistant",
            "content": self.content
        }

class ThinkingClient:
    """O1モデル用の思考プロセス専用クライアント"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.messages = []
        self._context = """
あなたは与えられた質問や情報を分析し、適切な回答を生成する思考エンジンです。

データベースには以下の情報が含まれています：
- products テーブル: 商品情報
- categories テーブル: カテゴリ情報

回答の際は以下の点に注意してください：
1. 情報が不足している場合は、「データベースの検索が必要です」と明確に示してください
2. データベースから情報を取得した場合は、その情報を分かりやすく整理して提示してください
3. 技術的な詳細は省略し、ユーザーにとって重要な情報に焦点を当ててください
4. タスクが完了した場合は、「タスクは完了しました」と明確に示し、実行結果の要約も含めてください
5. 追加の情報や確認が必要な場合は、「追加の情報が必要です」と明確に示してください

回答フォーマット：
---
【実行結果】
（実行した内容や取得したデータの要約）

【タスク状態】
タスクは完了しました。または追加の情報が必要です。

【詳細情報】
（データベースの構造や内容、追加したデータなどの詳細）
---
"""
        logger.info(f"ThinkingClient初期化完了: モデル={config.model}")
    
    async def think(self, context: str, tool_result: Optional[str] = None, iteration: int = 0) -> ThinkingResponse:
        """思考プロセスの実行"""
        logger.info(f"=== O1モデルの思考プロセス開始 (イテレーション: {iteration}) ===")
        if tool_result:
            prompt = f"""
{self._context}

ユーザーからの質問や状況:
{context}

データベースから取得した情報:
{tool_result}

この情報を分析し、以下の点を評価してください：
1. タスクは完全に完了したか
2. 追加の情報や確認が必要か
3. 現在の情報でユーザーに価値のある回答が可能か

これはイテレーション{iteration}回目です（最大2回の追加実行が可能）。
必要な場合は、「追加の情報が必要です」と明確に示してください。

分析結果を元に、ユーザーにとって分かりやすい回答を作成してください。
タスクが完了した場合は、「タスクは完了しました」と明確に示し、実行結果の要約も含めてください。
"""
            logger.info("ツール実行結果を含む思考プロセス")
        else:
            prompt = f"""
{self._context}

ユーザーからの質問:
{context}

この質問に答えるために必要な情報を分析してください。
これはイテレーション{iteration}回目の分析です。
データベースの検索や確認が必要な場合は、「データベースの検索が必要です」と明確に示してください。
"""
            logger.info("初期分析の思考プロセス")
            
        logger.debug(f"O1モデルへの入力プロンプト: {prompt}")
            
        try:
            logger.info("O1 APIリクエスト開始")
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_completion_tokens=32768  # O1モデルの推奨設定
            )
            logger.info("O1 APIリクエスト完了")
            logger.debug(f"O1モデルからの生の応答: {completion}")
            
            response = ThinkingResponse(completion)
            logger.info(f"思考プロセスの結果: ツール使用必要={response.needs_tool}, タスク完了={response.task_completed}")
            return response
            
        except Exception as e:
            logger.error(f"思考プロセスでエラー発生: {str(e)}")
            raise
