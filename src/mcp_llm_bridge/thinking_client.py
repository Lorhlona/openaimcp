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
        
    def _get_response_type(self) -> str:
        """応答タイプを判断"""
        if "【応答タイプ】" in self.content:
            response_type_line = [line for line in self.content.split('\n') if "【応答タイプ】" in line][0]
            return response_type_line.split("【応答タイプ】")[1].strip()
        return "ツール操作"  # デフォルトは従来の動作

    def _analyze_needs_tool(self) -> bool:
        """思考内容からツール使用の必要性を判断"""
        # タスクが完了している場合は常にFalse
        if self.task_completed:
            return False
            
        # 応答タイプに基づいて判断
        response_type = self._get_response_type()
        if response_type in ["会話応答", "ガイダンス"]:
            return False

        # ツール操作の場合は判断ロジック
        tool_indicators = [
            "データベースの検索が必要",
            "情報を取得する必要があります",
            "確認が必要です",
            "調べる必要があります",
            "追加の情報が必要です",
            "データベースを確認する必要があります",
            "検索が必要です",
            "インターネットで調べる必要があります",
            "オンライン情報が必要です",
            "最新情報を取得する必要があります",
            "【実行フェーズ】"  # フェーズ情報がある場合はツール使用が必要
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
            "確認が完了しました",
            "必要な情報が揃いました",
            "十分な情報が得られました"
        ]
        
        # 未完了を示す表現
        incomplete_indicators = [
            "追加の情報が必要です",
            "さらなる確認が必要です",
            "まだ完了していません",
            "もう少し調査が必要です",
            "情報が不十分です",
            "確認できていません",
            "さらなる検索が必要です",
            "追加の検索が必要です"
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
あなたは与えられた入力を分析し、適切な応答方法を判断する思考エンジンです。

# 入力の分類と応答
1. データベース操作が必要な場合:
   - 商品情報の検索や確認
   - カテゴリ情報の取得
   - データベース内の情報分析
   → データベースクエリを含む実行フェーズで応答

2. インターネット検索が必要な場合:
   - 一般的な情報収集
   - 最新情報の取得
   - オンラインリソースの検索
   → Google検索を含む実行フェーズで応答

3. 追加情報が必要な場合:
   - 質問の意図が不明確
   - 必要な情報が不足
   - 対象の特定が必要
   → 自然な対話形式でユーザーに質問

4. 複合的な操作が必要な場合:
   - データベース情報と外部情報の比較
   - 商品情報の市場調査
   - 競合分析
   → 複数のツールを組み合わせた実行フェーズで応答

4. 単純な会話の場合:
   - 挨拶、感謝
   - 一般的な雑談
   → 直接応答形式で返答

5. 空の入力や無意味な入力の場合:
   → 適切なガイダンスを提供

# 利用可能なツール情報
1. ユーザーとの対話 (human_interaction)
   - 自然な会話形式で質問
   - 具体的な情報を1つずつ確認
   - 質問の意図を明確に
   - 例: "どの写真集をお探しですか？"

2. データベース (database_query)
   - products テーブル: 商品情報
   - categories テーブル: カテゴリ情報

3. Google検索 (google_search)
   - クエリベースの検索
   - 最大10件の結果取得
   - 関連性の高い情報を優先

# 制約事項（ツール操作時）
1. ユーザーとの対話:
   - 1フェーズで1つの質問のみ
   - 自然な会話形式を維持
   - 質問は具体的かつ明確に
   - 情報不足時は最優先で使用

2. データベース/検索操作の場合:
   - 1フェーズで最大3つまでの操作
   - データベースとGoogle検索の組み合わせ可
   - 例: DBクエリ2回 + 検索1回

3. フェーズ制限:
   - 最大3フェーズまで（初回 + 2回の追加実行）
   - 各フェーズは明確な目的が必要
   - 対話フェーズは他のツールと分離

4. 検索制限:
   - 1回の検索で最大10件まで
   - 具体的で焦点を絞ったクエリを使用

# 応答フォーマット
## ツール操作が必要な場合:
---
【実行結果】
（実行内容や取得データの要約）

【タスク状態】
タスクは完了しました。または追加の情報が必要です。

【実行フェーズ】
現在のフェーズ: X/Y
実行すべき操作:
1. [Human_Interaction] 自然な対話形式での質問
または
1. [DB/検索] 目的1の実行
2. [DB/検索] 目的2の実行
3. [DB/検索] 目的3の実行

【詳細情報】
（具体的な実行計画や結果の詳細）
---

## 単純な会話の場合:
---
【応答タイプ】
会話応答

【応答内容】
（自然な会話形式での返答）
---

## 空/無意味な入力の場合:
---
【応答タイプ】
ガイダンス

【応答内容】
（適切なガイダンスメッセージ）
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

前回の実行結果:
{tool_result}

この情報を分析し、以下の点を評価してください：
1. タスクは完全に完了したか
2. 追加の情報や確認が必要か
   - データベースからの追加情報が必要か
   - インターネット検索による補完が必要か
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

この質問を以下のように分析してください：

1. 必要なツール操作を特定（最大3つまで）:
   - データベースクエリが必要か
   - Google検索が必要か
   - 両方のツールの組み合わせが効果的か
   - ユーザーとの対話が必要か

2. 各操作の目的を明確にする:
   - データベース操作の場合は具体的なクエリ内容
   - 検索操作の場合は検索キーワードと取得件数
   - 対話の場合は具体的な質問内容

3. 実行フェーズとして構造化する:
   - フェーズ番号と目的
   - 具体的な実行手順
   - 期待される結果

これはイテレーション{iteration}回目の分析です。
必ず【実行フェーズ】セクションを含めて回答してください。

例：
【実行結果】
商品情報の確認と市場調査が必要です。

【タスク状態】
追加の情報が必要です。

【実行フェーズ】
現在のフェーズ: 1/2
実行すべき操作:
1. [Human_Interaction] どの商品についての情報をお探しですか？
または
1. [DB] 商品情報の取得
2. [検索] 市場価格の調査
3. [DB] カテゴリ情報の確認

【詳細情報】
まずユーザーから具体的な情報を確認し、その後必要な情報を収集します。
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
