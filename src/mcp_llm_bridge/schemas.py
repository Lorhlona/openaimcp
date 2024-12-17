from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Operation(BaseModel):
    """単一の操作を表現するスキーマ"""
    type: str = Field(..., description="操作タイプ (human_interaction/database_query/google_search)")
    parameters: Dict[str, Any] = Field(..., description="操作のパラメータ")

class TaskPhase(BaseModel):
    """実行フェーズを表現するスキーマ"""
    phase_number: int = Field(..., description="フェーズ番号")
    operations: List[Operation] = Field(..., description="実行する操作のリスト（最大3つ）")
    description: str = Field(..., description="フェーズの説明")

class TaskPlan(BaseModel):
    """タスク全体の計画を表現するスキーマ"""
    overall_tasks: List[str] = Field(..., description="達成すべきサブタスクのリスト")
    total_phases: int = Field(..., description="予定される総フェーズ数")
    phases: List[TaskPhase] = Field(..., description="各フェーズの詳細")
    
class ThinkingResponse(BaseModel):
    """思考プロセスの応答を表現するスキーマ"""
    task_plan: Optional[TaskPlan] = Field(None, description="タスク計画（初回のみ）")
    current_phase: Optional[TaskPhase] = Field(None, description="現在のフェーズの実行計画")
    needs_tool: bool = Field(..., description="ツール使用の必要性")
    task_completed: bool = Field(..., description="タスクが完了したかどうか")
    final_response: Optional[str] = Field(None, description="最終的な応答（完了時）")

class ExecutionResult(BaseModel):
    """実行結果を表現するスキーマ"""
    operation_type: str = Field(..., description="実行された操作のタイプ")
    success: bool = Field(..., description="実行が成功したかどうか")
    result: Any = Field(..., description="実行結果")
    error: Optional[str] = Field(None, description="エラーメッセージ（失敗時）")