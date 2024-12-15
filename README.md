# MCP LLM Bridge

ユーザーの質問に対して、O1モデルとGPT-4を組み合わせて高度な思考プロセスを実現するAIシステムです。

## 主な機能

### 1. 思考プロセス
- O1モデルによる質問の分析と実行計画の立案
- GPT-4による具体的なツール実行
- 段階的な情報収集と結果の分析

### 2. 利用可能なツール
- **ユーザーとの対話** (human_interaction)
  - 自然な会話形式での質問
  - 具体的な情報の収集
  - 意図の明確化

- **Google検索** (google_search)
  - インターネットからの情報収集
  - 最大10件の関連結果取得
  - 具体的なキーワードによる検索

- **データベースクエリ** (database_query)
  - SQLiteデータベースへのクエリ実行
  - 商品情報やカテゴリ情報の取得
  - データの分析と集計

### 3. 使用例

```bash
Enter your prompt (or 'quit' to exit): AAの曲名なんだったっけな

🤖 AAとはどのアーティストやグループを指していますか？
👤 FF11のアークエンジェルつまりAAと戦うときの曲だよ

【実行結果】
FF11のアークエンジェル戦で流れる曲は「Fighters of the Crystal」です。
作曲者は水田直志氏です。
```

このように、ユーザーの曖昧な質問に対して：
1. まず質問で対話的に意図を明確化
2. 必要に応じてGoogle検索やデータベース検索を実行
3. 収集した情報を分析して最終的な回答を提供

## セットアップ

```bash
# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/bartolli/mcp-llm-bridge.git
cd mcp-llm-bridge
uv venv
source .venv/bin/activate  # Linux/Mac
# または
.\.venv\Scripts\Activate.ps1  # Windows
uv pip install -e .

# Create test database
python -m mcp_llm_bridge.create_test_db
```

## 設定

### OpenAI API設定

`.env`ファイルを作成：

```bash
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o  # Function Calling対応モデル
```

### Additional Endpoint Support

The bridge also works with any endpoint implementing the OpenAI API specification:

#### Ollama

```python
llm_config=LLMConfig(
    api_key="not-needed",
    model="mistral-nemo:12b-instruct-2407-q8_0",
    base_url="http://localhost:11434/v1"
)
```

Note: After testing various models, including `llama3.2:3b-instruct-fp16`, I found that `mistral-nemo:12b-instruct-2407-q8_0` handles complex queries more effectively.

#### LM Studio

```python
llm_config=LLMConfig(
    api_key="not-needed",
    model="local-model",
    base_url="http://localhost:1234/v1"
)
```

I didn't test this, but it should work.

### 実行

```bash
python -m mcp_llm_bridge.main
```

## ライセンス

[MIT](LICENSE.md)

## 貢献

PRs welcome.

cd mcp-llm-bridge ; .venv/Scripts/activate ; python -m mcp_llm_bridge.main


cd mcp-llm-bridge
.\.venv\Scripts\Activate.ps1
python -m mcp_llm_bridge.main