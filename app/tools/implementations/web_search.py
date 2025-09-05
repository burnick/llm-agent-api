"""Web search tool for retrieving information from the internet."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
from ..base import BaseTool, ToolResult, ToolExecutionError, ToolExecutionStatus
from ..context import ToolExecutionContext

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class WebSearchTool(BaseTool):
    """Tool for searching the web using a search API."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine: str = "duckduckgo"):
        """Initialize the web search tool.
        
        Args:
            api_key: API key for search service (if required)
            search_engine: Search engine to use ('duckduckgo', 'google', 'bing')
        """
        self.api_key = api_key
        self.search_engine = search_engine.lower()
        
        parameters_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to execute"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                },
                "safe_search": {
                    "type": "boolean",
                    "description": "Enable safe search filtering (default: true)",
                    "default": True
                }
            },
            "required": ["query"]
        }
        
        super().__init__(
            name="web_search",
            description="Search the web for information using a search engine. Returns titles, URLs, and snippets of relevant web pages.",
            parameters_schema=parameters_schema,
            timeout=30.0,
            required_permissions=["web_access"]
        )
    
    async def _search_duckduckgo(self, query: str, max_results: int, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            safe_search: Whether to enable safe search
            
        Returns:
            List of search results
        """
        if not AIOHTTP_AVAILABLE:
            raise ToolExecutionError(
                "aiohttp is required for web search functionality",
                error_code="DEPENDENCY_MISSING"
            )
        
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "safe_search": "strict" if safe_search else "moderate"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status != 200:
                        raise ToolExecutionError(
                            f"Search API returned status {response.status}",
                            error_code="API_ERROR"
                        )
                    
                    data = await response.json()
                    results = []
                    
                    # Process instant answer if available
                    if data.get("AbstractText"):
                        results.append({
                            "title": data.get("AbstractSource", "DuckDuckGo"),
                            "url": data.get("AbstractURL", ""),
                            "snippet": data.get("AbstractText", ""),
                            "type": "instant_answer"
                        })
                    
                    # Process related topics
                    for topic in data.get("RelatedTopics", [])[:max_results]:
                        if isinstance(topic, dict) and "Text" in topic:
                            results.append({
                                "title": topic.get("Text", "").split(" - ")[0] if " - " in topic.get("Text", "") else "Related Topic",
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", ""),
                                "type": "related_topic"
                            })
                    
                    return results[:max_results]
                    
            except asyncio.TimeoutError:
                raise ToolExecutionError(
                    "Search request timed out",
                    error_code="TIMEOUT"
                )
            except Exception as e:
                if AIOHTTP_AVAILABLE and hasattr(e, '__module__') and 'aiohttp' in e.__module__:
                    raise ToolExecutionError(
                        f"Network error during search: {str(e)}",
                        error_code="NETWORK_ERROR"
                    )
                else:
                    raise
    
    async def _search_fallback(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback search method when APIs are unavailable.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List with a single result indicating search is unavailable
        """
        return [{
            "title": "Search Unavailable",
            "url": "",
            "snippet": f"Web search is currently unavailable. Query was: {query}",
            "type": "error"
        }]
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        """Execute the web search tool.
        
        Args:
            parameters: Tool parameters containing 'query', 'max_results', and 'safe_search'
            context: Execution context for permission and domain checking
            
        Returns:
            ToolResult with search results
        """
        start_time = time.time()
        
        try:
            query = parameters["query"].strip()
            max_results = parameters.get("max_results", 5)
            safe_search = parameters.get("safe_search", True)
            
            if not query:
                raise ToolExecutionError(
                    "Search query cannot be empty",
                    error_code="EMPTY_QUERY"
                )
            
            # Check network permissions
            if context and not context.allow_network_access:
                return ToolResult(
                    status=ToolExecutionStatus.PERMISSION_DENIED,
                    error_message="Network access is not allowed in this context",
                    execution_time=time.time() - start_time,
                    metadata={"required_permission": "network_access"}
                )
            
            # Perform search based on configured engine
            try:
                if self.search_engine == "duckduckgo":
                    results = await self._search_duckduckgo(query, max_results, safe_search)
                else:
                    # For other search engines, use fallback for now
                    results = await self._search_fallback(query, max_results)
                
                execution_time = time.time() - start_time
                
                return ToolResult(
                    status=ToolExecutionStatus.SUCCESS,
                    result={
                        "query": query,
                        "results": results,
                        "total_results": len(results),
                        "search_engine": self.search_engine
                    },
                    execution_time=execution_time,
                    metadata={
                        "tool_version": "1.0",
                        "search_engine": self.search_engine,
                        "safe_search": safe_search
                    }
                )
                
            except ToolExecutionError:
                # Re-raise tool execution errors
                raise
            except Exception as e:
                # Fallback to error result for unexpected errors
                results = await self._search_fallback(query, max_results)
                execution_time = time.time() - start_time
                
                return ToolResult(
                    status=ToolExecutionStatus.ERROR,
                    result={
                        "query": query,
                        "results": results,
                        "total_results": len(results),
                        "search_engine": "fallback"
                    },
                    error_message=f"Search failed: {str(e)}",
                    execution_time=execution_time,
                    metadata={
                        "error_type": type(e).__name__,
                        "fallback_used": True
                    }
                )
                
        except ToolExecutionError:
            # Re-raise tool execution errors
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=f"Unexpected error: {str(e)}",
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )