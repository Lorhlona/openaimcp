import os
import tempfile
import logging
import requests
import pygame
import atexit
from typing import Optional

logger = logging.getLogger(__name__)

class VoiceManager:
    """音声出力を管理するクラス"""
    
    def __init__(self):
        """音声マネージャーの初期化"""
        self.api_base_url = os.getenv("NIJIVOICE_API_BASE_URL")
        self.voice_actor_id = os.getenv("VOICE_ACTOR_ID")
        self.api_key = os.getenv("NIJIVOICE_API_KEY")
        self.voice_mode = os.getenv("VOICE_MODE", "true").lower() == "true"
        self._temp_files = []  # 一時ファイルのリストを保持
        
        # 終了時に一時ファイルを削除するための登録
        atexit.register(self._cleanup_temp_files)
        
        # Pygameの初期化
        try:
            pygame.mixer.init()
            logger.info("Pygame mixer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pygame mixer: {str(e)}")
            raise

    def is_voice_enabled(self) -> bool:
        """音声出力が有効かどうかを確認"""
        return self.voice_mode and self.api_key is not None

    def process_text(self, text: str) -> Optional[str]:
        """テキストを音声に変換して再生"""
        if not self.is_voice_enabled():
            return None
            
        logger.debug(f"Processing text: {text}")
        logger.debug(f"Voice enabled: {self.is_voice_enabled()} (mode: {self.voice_mode}, api_key: {'set' if self.api_key else 'not set'})")
        
        try:
            # APIリクエストを送信
            url = "https://api.nijivoice.com/api/platform/v1/voice-actors/1fc717fe-ebf9-402b-9d8c-c59cda93d5dc/generate-voice"
            headers = {
                "x-api-key": self.api_key,
                "accept": "application/json",
                "content-type": "application/json"
            }
            payload = {
                "script": text,
                "speed": "1.0",
                "format": "mp3"
            }
            
            logger.debug(f"Sending request to {url}")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Payload: {payload}")
            
            response = requests.post(url, headers=headers, json=payload)
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response content type: {response.headers.get('content-type')}")
            
            if response.status_code == 200:
                response_json = response.json()
                logger.debug(f"JSON response: {response_json}")
                
                if audio_url := response_json.get("generatedVoice", {}).get("audioFileUrl"):
                    logger.debug(f"Found audio URL: {audio_url}")
                    
                    # 音声データをダウンロード
                    audio_response = requests.get(audio_url)
                    if audio_response.status_code == 200:
                        audio_data = audio_response.content
                        logger.debug("Successfully downloaded audio data")
                        
                        # データサイズの確認
                        if len(audio_data) > 0:
                            logger.debug("Data size check passed")
                            
                            # 一時ファイルを作成
                            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                            temp_file.write(audio_data)
                            temp_file.close()
                            self._temp_files.append(temp_file.name)  # 一時ファイルを記録
                            
                            logger.debug(f"Temporary file created at: {temp_file.name}")
                            
                            # 音声を再生
                            try:
                                pygame.mixer.music.load(temp_file.name)
                                pygame.mixer.music.play()
                                while pygame.mixer.music.get_busy():
                                    pygame.time.Clock().tick(10)
                            except Exception as e:
                                logger.error(f"Error playing audio: {str(e)}")
                                raise
                            
                text = text[:497] + "..."
            
            return audio_data
                
        except Exception as e:
            logger.error(f"Error generating voice: {str(e)}")
            return None
    
    def play_voice(self, voice_data: bytes) -> bool:
        """音声データを再生"""
        if not self.verify_mp3_data(voice_data):
            logger.error("Invalid MP3 data received")
            return False
            
        temp_file_path = None
        try:
            # 一時ファイルを作成して音声データを保存
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, mode='wb') as temp_file:
                temp_file.write(voice_data)
                temp_file_path = temp_file.name
            
            logger.debug(f"Temporary file created at: {temp_file_path}")
            
            # 既存の再生を停止
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
            # 音声を再生
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()
            
            # 再生が終わるまで待機
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # 再生完了後、少し待機してからファイルを削除
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing voice: {str(e)}")
            return False
            
        finally:
            # 一時ファイルを削除（リトライ付き）
            if temp_file_path and os.path.exists(temp_file_path):
                for _ in range(3):  # 最大3回リトライ
                    try:
                        os.unlink(temp_file_path)
                        logger.debug("Temporary file deleted successfully")
                        break
                    except Exception as e:
                        logger.error(f"Error deleting temporary file: {str(e)}")
                        time.sleep(0.5)  # 0.5秒待機してリトライ

    def _cleanup_temp_files(self):
        """プログラム終了時に一時ファイルを削除する"""
        for path in self._temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Failed to remove temp file {path}: {str(e)}")
        self._temp_files.clear()
