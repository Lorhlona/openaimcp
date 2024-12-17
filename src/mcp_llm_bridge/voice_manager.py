import os
import requests
import logging
import tempfile
import pygame
import time
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class VoiceManager:
    """にじボイスAPIを使用した音声出力管理クラス"""
    
    def __init__(self):
        self.api_key = os.getenv("NIJIVOICE_API_KEY")
        if not self.api_key:
            logger.warning("NIJIVOICE_API_KEY is not set. Voice output will be disabled.")
        
        # 音声モードの初期状態はオフ
        self.voice_mode = os.getenv("VOICE_MODE", "off").lower() == "on"
        
        # 固定の音声ID（モデルさん）
        self.voice_actor_id = os.getenv("VOICE_ACTOR_ID", "1fc717fe-ebf9-402b-9d8c-c59cda93d5dc")
        
        # APIエンドポイント
        self.api_base_url = os.getenv("NIJIVOICE_API_BASE_URL", "https://api.nijivoice.com/api/platform/v1")
        
        try:
            # pygameの初期化（より詳細な設定）
            pygame.mixer.quit()  # 既存のミキサーをクリーンアップ
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            logger.info("Pygame mixer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {str(e)}")

    def verify_mp3_data(self, data: bytes) -> bool:
        """MP3データの検証"""
        try:
            # MP3ヘッダーをチェック
            # ID3v2ヘッダー
            if data.startswith(b'ID3'):
                logger.debug("ID3v2 header detected")
                return True
            # MP3フレームヘッダー
            elif data.startswith(b'\xFF\xFB') or data.startswith(b'\xFF\xFA'):
                logger.debug("MP3 frame header detected")
                return True
            # データの長さをチェック
            elif len(data) > 1024:  # 最小サイズ
                logger.debug("Data size check passed")
                return True
            
            logger.error("MP3 header not found")
            return False
        except Exception as e:
            logger.error(f"Error verifying MP3 data: {str(e)}")
            return False
    
    def is_voice_enabled(self) -> bool:
        """音声出力が有効かどうかを確認"""
        enabled = self.voice_mode and self.api_key is not None
        logger.debug(f"Voice enabled: {enabled} (mode: {self.voice_mode}, api_key: {'set' if self.api_key else 'not set'})")
        return enabled
    
    def set_voice_mode(self, enabled: bool):
        """音声モードの切り替え"""
        self.voice_mode = enabled
        # 環境変数を更新（次回起動時も保持される）
        os.environ["VOICE_MODE"] = "on" if enabled else "off"
        logger.info(f"Voice mode {'enabled' if enabled else 'disabled'}")
    
    def process_text(self, text: str) -> bool:
        """テキストを処理して音声を生成し再生"""
        if not text:
            logger.warning("Empty text provided")
            return False
            
        logger.debug(f"Processing text: {text}")
        voice_data = self.generate_voice(text)
        if voice_data:
            if self.verify_mp3_data(voice_data):
                return self.play_voice(voice_data)
            else:
                logger.error("Generated data is not a valid MP3")
        return False
    
    def generate_voice(self, text: str) -> bytes | None:
        """テキストから音声を生成"""
        if not self.is_voice_enabled():
            logger.warning("Voice generation is disabled")
            return None
            
        if not text:
            logger.warning("Empty text provided for voice generation")
            return None
            
        try:
            # テキストを500文字以内に制限
            if len(text) > 500:
                text = text[:497] + "..."
            
            url = f"{self.api_base_url}/voice-actors/{self.voice_actor_id}/generate-voice"
            
            headers = {
                "x-api-key": self.api_key,
                "accept": "application/json",  # JSONレスポンスを要求
                "content-type": "application/json"
            }
            
            payload = {
                "script": text,
                "speed": "1.0",
                "format": "mp3"
            }
            
            logger.debug(f"Sending request to {url}")
            logger.debug(f"Headers: {json.dumps({k: v if k != 'x-api-key' else '[REDACTED]' for k, v in headers.items()}, indent=2)}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(url, json=payload, headers=headers)
            
            logger.debug(f"Response status code: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
            
            content_type = response.headers.get('content-type', '')
            logger.debug(f"Response content type: {content_type}")
            
            try:
                data = response.json()
                logger.debug(f"JSON response: {json.dumps(data, indent=2)}")
                
                if 'generatedVoice' in data and 'audioFileUrl' in data['generatedVoice']:
                    audio_url = data['generatedVoice']['audioFileUrl']
                    logger.debug(f"Found audio URL: {audio_url}")
                    
                    # 音声データをダウンロード
                    audio_response = requests.get(audio_url)
                    if audio_response.status_code == 200:
                        logger.debug("Successfully downloaded audio data")
                        return audio_response.content
                    else:
                        logger.error(f"Failed to download audio: {audio_response.status_code}")
                        return None
                else:
                    logger.error("No audio URL found in response")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                return None
                
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
