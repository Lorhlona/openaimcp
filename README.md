# MCP LLM Bridge

GPT-4とO1-miniを組み合わせた高度な対話システム。タスクを構造化し、複数のツールを組み合わせて効果的な情報検索と提供を実現します。

## 特徴

- O1-miniによる構造化された思考プロセス
- GPT-4による効率的なツール実行
- データベース、検索、対話の統合
- フェーズベースの実行管理
- 堅牢なエラーハンドリング

## アーキテクチャ

1. 思考エンジン（O1-mini）
   - タスクの分解と構造化
   - 実行計画の立案
   - 結果の分析

2. 実行エンジン（GPT-4）
   - データベースクエリ
   - Google検索
   - ユーザー対話

## 使用例

```
Enter your prompt: FF11のAAって敵キャラがいるんだけどそいつと戦うときのBGMの情報をしりたいの

🤖 具体的にどのような情報をお探しですか？
👤 うーんとね、FF11のAAって敵キャラがいるんだけどそいつと戦うときのBGMの情報をしりたいの

Response: FF11のAAとの戦闘時のBGMは「Fighters of the Crystal」です。詳細については以下のリンクをご参照ください。
1. [ヴァナ・ディール音楽紀行#2](https://www.youtube.com/watch?v=zh1oMAhjy-k)
2. [FFXI 戦闘曲集](https://www.youtube.com/playlist?list=PLDrtyx_shpKNMUzTdQqtyvebZBQ-QyVTm)
```

## 技術スタック

- Python 3.12+
- OpenAI API (GPT-4)
- O1-mini
- SQLite
- Pydantic
- asyncio/aiohttp

## 起動方法

Windows:
```bash
cd mcp-llm-bridge
.\.venv\Scripts\activate
python -m mcp_llm_bridge.main
```

Linux/Mac:
```bash
cd mcp-llm-bridge; source .venv/bin/activate; python -m mcp_llm_bridge.main
```

## ライセンス

MIT

## 貢献

PRs welcome