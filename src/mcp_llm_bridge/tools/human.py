from typing import Dict, Any
import json
import asyncio
from dataclasses import dataclass

@dataclass
class HumanToolResponse:
    """人間からの応答を格納するデータクラス"""
    answer: str
    success: bool = True
    error: str = ""

class HumanTool:
    """人間とのインタラクションを管理するMCPツール"""
    
    def __init__(self):
        self.name = "human_interaction"
        self.description = "ユーザーに追加の質問をして情報を収集するツール"
        
    def get_tool_spec(self) -> Dict[str, Any]:
        """ツールの仕様を返す"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
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
        
    async def execute(self, args: Dict[str, Any]) -> str:
        """ツールを実行し、ユーザーからの応答を待つ"""
        try:
            # パラメータの検証
            if 'question' not in args or not args['question'].strip():
                raise ValueError("質問が指定されていないか、空の質問です")

            # チャットスタイルで質問を表示
            print("\n🤖 " + args['question'])
            print("👤 ", end='', flush=True)
            
            # ユーザーからの入力を待つ
            response = await self._get_user_input()
            
            # 空の回答をチェック
            if not response.strip():
                raise ValueError("回答が入力されていません")
            
            # 応答をJSON形式で返す
            result = HumanToolResponse(answer=response)
            return json.dumps({
                "result": {
                    "answer": result.answer,
                    "success": result.success
                }
            }, ensure_ascii=False)
            
        except Exception as e:
            error_result = HumanToolResponse(
                answer="",
                success=False,
                error=str(e)
            )
            return json.dumps({
                "answer": error_result.answer,
                "success": error_result.success,
                "error": error_result.error
            }, ensure_ascii=False)
    
    async def _get_user_input(self) -> str:
        """非同期でユーザー入力を取得"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input)