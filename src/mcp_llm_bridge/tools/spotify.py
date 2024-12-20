import os
import logging
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import json

logger = logging.getLogger(__name__)

@dataclass
class SpotifyTool:
    """Spotifyの操作を行うツール"""
    
    def __init__(self):
        """SpotifyクライアントとOAuth認証の初期化"""
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        if not self.client_id:
            raise ValueError("SPOTIFY_CLIENT_ID environment variable is not set")
            
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not self.client_secret:
            raise ValueError("SPOTIFY_CLIENT_SECRET environment variable is not set")
            
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")
        
        self.scope = [
            "user-read-playback-state",
            "user-modify-playback-state",
            "user-read-currently-playing",
            "playlist-read-private",
            "playlist-read-collaborative",
            "playlist-modify-private",
            "playlist-modify-public",
            "user-library-read",
            "streaming",
            "app-remote-control"
        ]
        
        try:
            auth_manager = SpotifyOAuth(
                scope=' '.join(self.scope),
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                open_browser=True
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("Spotify client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {str(e)}")
            raise

    def get_tool_spec(self) -> Dict[str, Any]:
        """ツールの仕様を返す"""
        return {
            "name": "spotify",
            "description": "Spotifyの操作を行うツール",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "play", "pause", "current_track", "add_to_queue"],
                        "description": "実行するアクション"
                    },
                    "query": {
                        "type": "string",
                        "description": "検索クエリ（searchアクション用）"
                    },
                    "track_id": {
                        "type": "string",
                        "description": "トラックID（play/add_to_queueアクション用）"
                    }
                },
                "required": ["action"]
            }
        }

    def get_devices(self) -> List[Dict[str, Any]]:
        """利用可能なデバイスのリストを取得"""
        try:
            devices = self.sp.devices()
            return devices.get('devices', [])
        except Exception as e:
            logger.error(f"Error getting devices: {str(e)}")
            return []

    def get_active_device(self) -> Optional[Dict[str, Any]]:
        """アクティブなデバイスを取得"""
        devices = self.get_devices()
        return next((d for d in devices if d['is_active']), None)

    def get_best_device(self) -> Optional[Dict[str, Any]]:
        """最適なデバイスを取得"""
        devices = self.get_devices()
        if not devices:
            return None

        # アクティブなデバイスを優先
        active_device = self.get_active_device()
        if active_device:
            return active_device

        # 最初のデバイスを使用
        return devices[0]

    def ensure_device_ready(self) -> Optional[Dict[str, Any]]:
        """デバイスの準備を確認"""
        device = self.get_best_device()
        if not device:
            return None

        try:
            # デバイスをアクティブ化
            self.sp.transfer_playback(
                device_id=device['id'],
                force_play=True  # 強制的にアクティブ化
            )
            time.sleep(1)  # アクティベーションの待機
            return device
        except Exception as e:
            logger.error(f"Error activating device: {str(e)}")
            return None

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """ツールを実行する"""
        action = parameters.get("action")
        
        try:
            match action:
                case "search":
                    query = parameters.get("query")
                    if not query:
                        raise ValueError("search action requires query parameter")
                    results = self.sp.search(q=query, limit=5)
                    tracks = []
                    for track in results["tracks"]["items"]:
                        tracks.append({
                            "id": track["id"],
                            "name": track["name"],
                            "artist": track["artists"][0]["name"],
                            "album": track["album"]["name"],
                            "url": track["external_urls"]["spotify"]
                        })
                    return {"tracks": tracks}

                case "play":
                    track_id = parameters.get("track_id")
                    if not track_id:
                        raise ValueError("play action requires track_id parameter")

                    # デバイスの準備
                    device = self.ensure_device_ready()
                    if not device:
                        return {
                            "error": "デバイスが見つかりません。Spotifyアプリを開いて、デバイスを有効にしてください。",
                            "status": "device_not_found"
                        }

                    try:
                        # 再生を開始
                        self.sp.start_playback(
                            device_id=device['id'],
                            uris=[f"spotify:track:{track_id}"]
                        )
                        
                        # 再生状態を確認
                        time.sleep(1)
                        current = self.sp.current_playback()
                        if current and current.get('is_playing'):
                            return {
                                "status": "playing",
                                "track_id": track_id,
                                "device": device['name']
                            }
                        else:
                            return {
                                "error": "再生の開始を確認できませんでした",
                                "status": "playback_failed"
                            }
                    except Exception as e:
                        logger.error(f"Error starting playback: {str(e)}")
                        return {
                            "error": f"再生の開始に失敗しました: {str(e)}",
                            "status": "playback_error"
                        }

                case "pause":
                    try:
                        device = self.get_active_device()
                        if not device:
                            return {"error": "No active device available"}
                        
                        self.sp.pause_playback(device_id=device['id'])
                        return {"status": "paused"}
                    except Exception as e:
                        logger.error(f"Error pausing playback: {str(e)}")
                        return {"error": f"Failed to pause playback: {str(e)}"}

                case "current_track":
                    current = self.sp.current_user_playing_track()
                    if not current or not current.get("item"):
                        return {"status": "no_track_playing"}
                    track = current["item"]
                    return {
                        "status": "playing" if current["is_playing"] else "paused",
                        "track": {
                            "id": track["id"],
                            "name": track["name"],
                            "artist": track["artists"][0]["name"],
                            "album": track["album"]["name"],
                            "url": track["external_urls"]["spotify"]
                        }
                    }

                case "add_to_queue":
                    track_id = parameters.get("track_id")
                    if not track_id:
                        raise ValueError("add_to_queue action requires track_id parameter")
                    
                    device = self.get_active_device()
                    if not device:
                        return {"error": "No active device available"}

                    try:
                        self.sp.add_to_queue(
                            uri=f"spotify:track:{track_id}",
                            device_id=device['id']
                        )
                        return {
                            "status": "added_to_queue",
                            "track_id": track_id,
                            "device": device['name']
                        }
                    except Exception as e:
                        logger.error(f"Error adding to queue: {str(e)}")
                        return {"error": f"Failed to add to queue: {str(e)}"}

                case _:
                    raise ValueError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error executing Spotify action {action}: {str(e)}")
            return {"error": str(e)}
