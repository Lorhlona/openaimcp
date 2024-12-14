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
        self.needs_tool = self._analyze_needs_tool()
        logger.debug(f"生のレスポンス: {completion}")
        logger.debug(f"処理済みコンテンツ: {self.content}")
        
    def _analyze_needs_tool(self) -> bool:
        """思考内容からツール使用の必要性を判断"""
        tool_indicators = [
            "データベース",
            "クエリ",
            "検索",
            "情報を取得",
            "確認が必要",
            "調べる必要"
        ]
        needs_tool = any(indicator in self.content.lower() for indicator in tool_indicators)
        logger.debug(f"ツール使用の必要性判断: {needs_tool}")
        return needs_tool

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
1. 情報が不足している場合は、データベースの検索が必要なことを示してください
2. データベースから情報を取得した場合は、その情報を分かりやすく整理して提示してください
3. 技術的な詳細は省略し、ユーザーにとって重要な情報に焦点を当ててください
"""
        logger.info(f"ThinkingClient初期化完了: モデル={config.model}")
    
    async def think(self, context: str, tool_result: Optional[str] = None) -> ThinkingResponse:
        """思考プロセスの実行"""
        logger.info("=== O1モデルの思考プロセス開始 ===")
        if tool_result:
            prompt = f"""
{self._context}

ユーザーからの質問や状況:
{context}

データベースから取得した情報:
{tool_result}

この情報を分析し、ユーザーにとって分かりやすい回答を作成してください。
"""
            logger.info("ツール実行結果を含む思考プロセス")
        else:
            prompt = f"""
{self._context}

ユーザーからの質問:
{context}

この質問に答えるために必要な情報を分析してください。
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
            logger.info(f"思考プロセスの結果: ツール使用必要={response.needs_tool}")
            return response
            
        except Exception as e:
            logger.error(f"思考プロセスでエラー発生: {str(e)}")
            raise
