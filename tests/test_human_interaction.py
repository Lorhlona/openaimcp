import pytest
from unittest.mock import Mock, patch
from mcp_llm_bridge.thinking_client import ThinkingResponse, ThinkingClient
from mcp_llm_bridge.bridge import MCPLLMBridge
from mcp_llm_bridge.config import BridgeConfig, LLMConfig
import json

@pytest.fixture
def mock_completion():
    completion = Mock()
    completion.choices = [Mock()]
    completion.choices[0].message.content = """
【応答タイプ】
人間への質問

【質問内容】
なっちゃんとはだれですか？

【コンテキスト保持】
元の質問: なっちゃんの写真集の値段を教えて
必要な情報: 人物の特定
次のステップ: 特定された人物の写真集の価格をGoogle検索で調査
"""
    return completion

def test_thinking_response_human_input_detection(mock_completion):
    response = ThinkingResponse(mock_completion)
    assert response.needs_human_input == True
    assert response._get_response_type() == "人間への質問"
    assert "なっちゃんとはだれですか？" in response.get_human_question()

@pytest.mark.asyncio
async def test_bridge_human_interaction():
    # モックの設定
    config = BridgeConfig(
        mcp_server_params=None,
        llm_config=LLMConfig(
            api_key="test",
            model="test",
            base_url=None
        )
    )
    
    bridge = MCPLLMBridge(config)
    
    # 最初の質問
    with patch('builtins.input', return_value="安倍なつみです"):
        response = await bridge.process_message("なっちゃんの写真集の値段おしえて")
        
    # 応答に質問が含まれていることを確認
    assert "なっちゃん" in response
    assert "だれですか" in response

    # コンテキストが保持されていることを確認
    assert "安倍なつみ" in bridge.context_memory.get("human_answer", "")