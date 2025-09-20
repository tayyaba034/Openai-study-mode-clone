"""
MCP (Model Context Protocol) Server implementation for the Study Mode application.
Handles communication with external services and manages agent context.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages in the MCP protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"

@dataclass
class MCPMessage:
    """Represents an MCP message."""
    message_type: MessageType
    message_id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class MCPServer:
    """
    MCP Server for handling agent communication and external service integration.
    """
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.message_handlers: Dict[str, callable] = {}
        self.context_store: Dict[str, Any] = {}
        self.setup_default_handlers()
    
    def setup_default_handlers(self):
        """Setup default message handlers."""
        self.message_handlers.update({
            "register_agent": self._handle_register_agent,
            "execute_task": self._handle_execute_task,
            "get_context": self._handle_get_context,
            "update_context": self._handle_update_context,
            "list_agents": self._handle_list_agents,
            "agent_status": self._handle_agent_status
        })
    
    async def _handle_register_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent registration."""
        agent_id = params.get("agent_id")
        agent_info = params.get("agent_info", {})
        
        if not agent_id:
            return {"error": "Agent ID is required"}
        
        # Update existing agent or create new one
        if agent_id in self.agents:
            self.agents[agent_id]["last_seen"] = asyncio.get_event_loop().time()
        else:
            self.agents[agent_id] = {
                "info": agent_info,
                "status": "active",
                "last_seen": asyncio.get_event_loop().time(),
                "capabilities": agent_info.get("capabilities", [])
            }
        
        logger.info(f"Agent {agent_id} registered successfully")
        return {"success": True, "agent_id": agent_id}
    
    async def _handle_execute_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task execution requests."""
        agent_id = params.get("agent_id")
        task = params.get("task")
        context = params.get("context", {})
        
        if not agent_id or not task:
            return {"error": "Agent ID and task are required"}
        
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}
        
        # Store task context
        task_id = f"{agent_id}_{asyncio.get_event_loop().time()}"
        self.context_store[task_id] = context
        
        try:
            # This would typically route to the specific agent
            result = await self._route_to_agent(agent_id, task, context)
            return {"success": True, "task_id": task_id, "result": result}
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {"error": str(e)}
    
    async def _handle_get_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context retrieval requests."""
        task_id = params.get("task_id")
        agent_id = params.get("agent_id")
        
        if task_id and task_id in self.context_store:
            return {"context": self.context_store[task_id]}
        elif agent_id and agent_id in self.agents:
            return {"context": self.agents[agent_id].get("context", {})}
        else:
            return {"error": "Context not found"}
    
    async def _handle_update_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle context updates."""
        task_id = params.get("task_id")
        context_update = params.get("context", {})
        
        if not task_id:
            return {"error": "Task ID is required"}
        
        if task_id in self.context_store:
            self.context_store[task_id].update(context_update)
            return {"success": True}
        else:
            return {"error": "Task ID not found"}
    
    async def _handle_list_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent listing requests."""
        return {
            "agents": [
                {
                    "id": agent_id,
                    "status": agent["status"],
                    "capabilities": agent["capabilities"]
                }
                for agent_id, agent in self.agents.items()
            ]
        }
    
    async def _handle_agent_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent status requests."""
        agent_id = params.get("agent_id")
        
        if not agent_id:
            return {"error": "Agent ID is required"}
        
        if agent_id in self.agents:
            return {
                "agent_id": agent_id,
                "status": self.agents[agent_id]["status"],
                "last_seen": self.agents[agent_id]["last_seen"]
            }
        else:
            return {"error": "Agent not found"}
    
    async def _route_to_agent(self, agent_id: str, task: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Route task execution to the appropriate agent."""
        # This is a placeholder - in a real implementation, this would
        # route to the actual agent implementation
        logger.info(f"Routing task to agent {agent_id}: {task}")
        return {"status": "completed", "agent_id": agent_id}
    
    async def send_message(self, message: MCPMessage) -> MCPMessage:
        """Send an MCP message and return the response."""
        try:
            if message.method in self.message_handlers:
                handler = self.message_handlers[message.method]
                result = await handler(message.params or {})
                
                response = MCPMessage(
                    message_type=MessageType.RESPONSE,
                    message_id=message.message_id,
                    result=result
                )
                return response
            else:
                error_response = MCPMessage(
                    message_type=MessageType.ERROR,
                    message_id=message.message_id,
                    error={"code": -32601, "message": "Method not found"}
                )
                return error_response
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            error_response = MCPMessage(
                message_type=MessageType.ERROR,
                message_id=message.message_id,
                error={"code": -32603, "message": str(e)}
            )
            return error_response
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register a new agent with the MCP server."""
        # Register synchronously during initialization
        self.agents[agent_id] = {
            "info": agent_info,
            "status": "active",
            "last_seen": 0,  # Will be updated when first used
            "capabilities": agent_info.get("capabilities", [])
        }
        logger.info(f"Agent {agent_id} registered successfully")
    
    def get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities of a registered agent."""
        if agent_id in self.agents:
            return self.agents[agent_id]["capabilities"]
        return []
    
    def is_agent_active(self, agent_id: str) -> bool:
        """Check if an agent is active."""
        return agent_id in self.agents and self.agents[agent_id]["status"] == "active"

# Global MCP server instance
mcp_server = MCPServer()
