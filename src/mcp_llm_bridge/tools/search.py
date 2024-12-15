"""
Google Search tool for MCP LLM Bridge using SerpAPI.
Provides functionality to perform Google searches and return formatted results.
"""

from typing import Dict, List, Any
import os
import logging
import aiohttp
import json
from urllib.parse import urlencode

class GoogleSearchTool:
    """Tool for performing Google searches using SerpAPI"""
    
    def __init__(self):
        self.api_key = os.getenv("SERPAPI_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY environment variable is required")
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://serpapi.com/search"
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """Get the tool specification in MCP format"""
        return {
            "name": "google_search",
            "description": "Perform a Google search and get relevant results",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (max 10)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    
    async def execute(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a Google search and return results"""
        query = params.get("query")
        if not query:
            raise ValueError("Query parameter is required")
            
        num_results = min(params.get("num_results", 5), 10)
        
        search_params = {
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
            "engine": "google"
        }
        
        url = f"{self.base_url}?{urlencode(search_params)}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"SerpAPI request failed: {error_text}")
                    
                    data = await response.json()
                    
                    if "error" in data:
                        raise ValueError(f"SerpAPI error: {data['error']}")
                    
                    organic_results = data.get("organic_results", [])
                    formatted_results = []
                    
                    for result in organic_results[:num_results]:
                        formatted_results.append({
                            "title": result.get("title"),
                            "link": result.get("link"),
                            "snippet": result.get("snippet"),
                            "position": result.get("position")
                        })
                    
                    return formatted_results
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error during search: {str(e)}")
            raise ValueError(f"Network error during search: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse SerpAPI response: {str(e)}")
            raise ValueError(f"Failed to parse search results: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error during search: {str(e)}")
            raise ValueError(f"Search failed: {str(e)}")