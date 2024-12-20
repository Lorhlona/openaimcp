# src/mcp_llm_bridge/main.py
import os
import asyncio
from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp_llm_bridge.config import BridgeConfig, LLMConfig
from mcp_llm_bridge.bridge import BridgeManager
import colorlog
import logging

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
logger.setLevel(logging.INFO)

async def main():
    # 環境変数の読み込み
    load_dotenv()

    # プロジェクトルートディレクトリの取得
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    db_path = os.path.join(project_root, "test.db")
    
    # ブリッジの設定
    config = BridgeConfig(
        mcp_server_params=StdioServerParameters(
            command="uvx",
            args=["mcp-server-sqlite", "--db-path", db_path],
            env=None
        ),
        # ツール実行用の設定（Function Calling対応モデル）
        llm_config=LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",  # Function Calling対応モデル
            base_url=None,
            max_tokens=2000
        ),
        # 思考プロセス用の設定（O1モデル）
        thinking_config=LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="o1-mini",  # O1モデル
            base_url=None,
            max_tokens=32768  # O1モデルの推奨設定
        )
    )
    
    logger.info(f"Starting bridge with thinking model: {config.thinking_config.model}")
    logger.info(f"Tool execution model: {config.llm_config.model}")
    logger.info(f"Using database at: {db_path}")
    
    # コンテキストマネージャーを使用してブリッジを実行
    async with BridgeManager(config) as bridge:
        iteration_count = 1

        while True:
            current_summary = bridge.summarize_context()
            logger.info("== 要約 ==")
            logger.info(current_summary)
            print("== 要約 ==")
            print(current_summary)

            user_input = input("\nEnter your prompt (or 'quit' to exit): ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            combined_input = f"{current_summary}\n\nユーザーの入力: {user_input}"

            logger.info(f"=== Iteration {iteration_count} start ===")

            response = await bridge.process_message(combined_input)

            logger.info(f"=== Iteration {iteration_count} completed ===")
            logger.info(f"Got response: {response}")

            print(f"\nResponse: {response}")

            if not bridge.is_task_completed:
                new_summary = bridge.summarize_context()
                bridge.context_summary = new_summary
            else:
                break

            iteration_count += 1

if __name__ == "__main__":
    asyncio.run(main())
