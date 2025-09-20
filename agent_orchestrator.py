"""
Agent Orchestrator for implementing Mixture of Experts (MoE) approach.
This module coordinates multiple AI agents and routes tasks to the most appropriate agents.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Import our agents
from agents.data_agent import DataProcessingAgent
from agents.preprocessing_agent import DataPreprocessingAgent
from agents.model_suggestion_agent import ModelSuggestionAgent
from agents.inference_agent import ParallelInferenceAgent
from mcp_server import mcp_server

logger = logging.getLogger(__name__)

class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

@dataclass
class TaskContext:
    """Context information for a task."""
    user_id: str
    session_id: str
    task_type: str
    complexity: TaskComplexity
    input_data: Any
    constraints: Optional[Dict[str, Any]] = None
    priority: int = 1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AgentOrchestrator:
    """
    Orchestrator for managing multiple AI agents using Mixture of Experts approach.
    Routes tasks to appropriate agents and coordinates their execution.
    """
    
    def __init__(self):
        # Initialize agents
        self.data_agent = DataProcessingAgent()
        self.preprocessing_agent = DataPreprocessingAgent()
        self.model_suggestion_agent = ModelSuggestionAgent()
        self.inference_agent = ParallelInferenceAgent()
        
        # Agent registry
        self.agents = {
            "data_processor": self.data_agent,
            "data_preprocessor": self.preprocessing_agent,
            "model_suggester": self.model_suggestion_agent,
            "parallel_inference": self.inference_agent
        }
        
        # Task routing rules
        self.routing_rules = {
            "data_processing": ["data_processor"],
            "data_cleaning": ["data_preprocessor"],
            "model_recommendation": ["model_suggester"],
            "parallel_inference": ["parallel_inference"],
            "study_question": ["model_suggester", "parallel_inference"],
            "data_analysis": ["data_processor", "data_preprocessor", "parallel_inference"],
            "text_conversion": ["data_processor"],
            "model_comparison": ["model_suggester", "parallel_inference"]
        }
        
        # Register agents with MCP server
        self._register_agents()
        
        # Task history and performance tracking
        self.task_history = []
        self.performance_metrics = {}
    
    def _register_agents(self):
        """Register all agents with the MCP server."""
        for agent_id, agent in self.agents.items():
            agent_info = {
                "id": agent_id,
                "type": agent.__class__.__name__,
                "capabilities": agent.get_capabilities(),
                "status": "active"
            }
            mcp_server.register_agent(agent_id, agent_info)
            logger.info(f"Registered agent: {agent_id}")
    
    async def process_study_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a study-related request using the MoE approach.
        
        Args:
            user_input: User's input/question
            context: Optional context information
            
        Returns:
            Comprehensive response from multiple agents
        """
        try:
            # Analyze the request to determine routing
            task_context = await self._analyze_request(user_input, context)
            
            # Route to appropriate agents
            selected_agents = await self._route_request(task_context)
            
            # Execute tasks in parallel
            results = await self._execute_parallel_tasks(task_context, selected_agents)
            
            # Combine and synthesize results
            final_response = await self._synthesize_results(results, task_context)
            
            # Update performance metrics
            self._update_metrics(task_context, results, final_response)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing study request: {e}")
            return {"error": str(e), "success": False}
    
    async def _analyze_request(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> TaskContext:
        """
        Analyze user request to determine task characteristics.
        
        Args:
            user_input: User's input
            context: Optional context
            
        Returns:
            TaskContext with analyzed information
        """
        try:
            # Use AI to analyze the request
            analysis_prompt = f"""
            Analyze this study request and determine:
            
            User Input: {user_input}
            Context: {json.dumps(context, indent=2) if context else "None"}
            
            Please analyze and return:
            1. Task type (data_processing, study_question, analysis, etc.)
            2. Complexity level (simple, moderate, complex, expert)
            3. Required capabilities
            4. Estimated processing time
            5. Priority level
            
            Return as JSON:
            {{
                "task_type": "type",
                "complexity": "simple|moderate|complex|expert",
                "required_capabilities": ["cap1", "cap2", ...],
                "estimated_time": "short|medium|long",
                "priority": 1-5,
                "keywords": ["keyword1", "keyword2", ...],
                "suggested_agents": ["agent1", "agent2", ...]
            }}
            """
            
            model = self.model_suggestion_agent.model
            response = model.generate_content(analysis_prompt)
            analysis = json.loads(response.text)
            
            # Create task context
            task_context = TaskContext(
                user_id=context.get("user_id", "anonymous") if context else "anonymous",
                session_id=context.get("session_id", f"session_{int(datetime.now().timestamp())}") if context else f"session_{int(datetime.now().timestamp())}",
                task_type=analysis.get("task_type", "study_question"),
                complexity=TaskComplexity(analysis.get("complexity", "moderate")),
                input_data=user_input,
                constraints=context,
                priority=analysis.get("priority", 1)
            )
            
            logger.info(f"Request analyzed: {task_context.task_type} - {task_context.complexity.value}")
            return task_context
            
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            # Fallback to default context
            return TaskContext(
                user_id="anonymous",
                session_id=f"session_{int(datetime.now().timestamp())}",
                task_type="study_question",
                complexity=TaskComplexity.MODERATE,
                input_data=user_input,
                constraints=context
            )
    
    async def _route_request(self, task_context: TaskContext) -> List[str]:
        """
        Route request to appropriate agents based on task context.
        
        Args:
            task_context: Analyzed task context
            
        Returns:
            List of selected agent IDs
        """
        try:
            selected_agents = []
            
            # Get routing rules for task type
            if task_context.task_type in self.routing_rules:
                base_agents = self.routing_rules[task_context.task_type]
                selected_agents.extend(base_agents)
            else:
                # Default routing for unknown task types
                selected_agents = ["parallel_inference"]
            
            # Adjust selection based on complexity
            if task_context.complexity == TaskComplexity.SIMPLE:
                # For simple tasks, use fewer agents
                selected_agents = selected_agents[:1] if selected_agents else ["parallel_inference"]
            elif task_context.complexity == TaskComplexity.EXPERT:
                # For expert tasks, use all relevant agents
                if "parallel_inference" not in selected_agents:
                    selected_agents.append("parallel_inference")
            
            # Ensure we have at least one agent
            if not selected_agents:
                selected_agents = ["parallel_inference"]
            
            logger.info(f"Routed to agents: {selected_agents}")
            return selected_agents
            
        except Exception as e:
            logger.error(f"Error routing request: {e}")
            return ["parallel_inference"]
    
    async def _execute_parallel_tasks(self, task_context: TaskContext, selected_agents: List[str]) -> Dict[str, Any]:
        """
        Execute tasks on selected agents in parallel.
        
        Args:
            task_context: Task context
            selected_agents: List of selected agent IDs
            
        Returns:
            Results from all agents
        """
        try:
            # Prepare tasks for each agent
            tasks = []
            for agent_id in selected_agents:
                if agent_id in self.agents:
                    task = self._prepare_agent_task(agent_id, task_context)
                    tasks.append(task)
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = {}
            for i, result in enumerate(results):
                agent_id = selected_agents[i]
                if isinstance(result, Exception):
                    processed_results[agent_id] = {
                        "success": False,
                        "error": str(result),
                        "agent_id": agent_id
                    }
                else:
                    processed_results[agent_id] = result
            
            logger.info(f"Executed tasks on {len(selected_agents)} agents")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error executing parallel tasks: {e}")
            return {"error": str(e)}
    
    async def _prepare_agent_task(self, agent_id: str, task_context: TaskContext) -> Dict[str, Any]:
        """
        Prepare task for a specific agent.
        
        Args:
            agent_id: Agent identifier
            task_context: Task context
            
        Returns:
            Task execution result
        """
        try:
            agent = self.agents[agent_id]
            
            # Create task based on agent type and task context
            task = {
                "type": self._determine_task_type(agent_id, task_context),
                "params": {
                    "input_data": task_context.input_data,
                    "context": task_context.constraints,
                    "complexity": task_context.complexity.value
                }
            }
            
            # Execute task
            result = await agent.execute_task(task)
            result["agent_id"] = agent_id
            result["execution_time"] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error preparing task for {agent_id}: {e}")
            return {"success": False, "error": str(e), "agent_id": agent_id}
    
    def _determine_task_type(self, agent_id: str, task_context: TaskContext) -> str:
        """
        Determine the specific task type for an agent.
        
        Args:
            agent_id: Agent identifier
            task_context: Task context
            
        Returns:
            Task type string
        """
        task_type_mapping = {
            "data_processor": {
                "study_question": "text_to_dataframe",
                "data_analysis": "extract_tabular_data",
                "text_conversion": "text_to_dataframe"
            },
            "data_preprocessor": {
                "study_question": "analyze_data_quality",
                "data_analysis": "clean_data",
                "data_cleaning": "clean_data"
            },
            "model_suggester": {
                "study_question": "analyze_task_requirements",
                "model_recommendation": "recommend_models",
                "model_comparison": "compare_models"
            },
            "parallel_inference": {
                "study_question": "parallel_inference",
                "data_analysis": "ensemble_inference",
                "model_comparison": "parallel_inference"
            }
        }
        
        agent_tasks = task_type_mapping.get(agent_id, {})
        return agent_tasks.get(task_context.task_type, "parallel_inference")
    
    async def _synthesize_results(self, results: Dict[str, Any], task_context: TaskContext) -> Dict[str, Any]:
        """
        Synthesize results from multiple agents into a coherent response.
        
        Args:
            results: Results from all agents
            task_context: Original task context
            
        Returns:
            Synthesized final response
        """
        try:
            # Collect successful results
            successful_results = {}
            for agent_id, result in results.items():
                if result.get("success", False):
                    successful_results[agent_id] = result.get("result", {})
            
            if not successful_results:
                return {
                    "success": False,
                    "error": "No agents completed successfully",
                    "results": results
                }
            
            # Use AI to synthesize results
            synthesis_prompt = f"""
            Synthesize these results from multiple AI agents into a coherent, comprehensive response:
            
            Original Request: {task_context.input_data}
            Task Type: {task_context.task_type}
            Complexity: {task_context.complexity.value}
            
            Agent Results:
            {json.dumps(successful_results, indent=2)}
            
            Please:
            1. Combine the best insights from each agent
            2. Create a unified, comprehensive response
            3. Maintain accuracy and avoid contradictions
            4. Structure the response clearly
            5. Include relevant details and examples
            
            Return as JSON:
            {{
                "synthesized_response": "comprehensive response text",
                "key_insights": ["insight1", "insight2", ...],
                "sources": ["agent1", "agent2", ...],
                "confidence": 0.95,
                "recommendations": ["rec1", "rec2", ...]
            }}
            """
            
            model = self.model_suggestion_agent.model
            response = model.generate_content(synthesis_prompt)
            synthesis = json.loads(response.text)
            
            final_response = {
                "success": True,
                "response": synthesis.get("synthesized_response", ""),
                "key_insights": synthesis.get("key_insights", []),
                "sources": synthesis.get("sources", []),
                "confidence": synthesis.get("confidence", 0.0),
                "recommendations": synthesis.get("recommendations", []),
                "agent_results": successful_results,
                "task_context": {
                    "task_type": task_context.task_type,
                    "complexity": task_context.complexity.value,
                    "timestamp": task_context.timestamp.isoformat()
                }
            }
            
            logger.info(f"Results synthesized from {len(successful_results)} agents")
            return final_response
            
        except Exception as e:
            logger.error(f"Error synthesizing results: {e}")
            return {
                "success": False,
                "error": str(e),
                "raw_results": results
            }
    
    def _update_metrics(self, task_context: TaskContext, results: Dict[str, Any], final_response: Dict[str, Any]):
        """Update performance metrics."""
        try:
            task_record = {
                "timestamp": task_context.timestamp.isoformat(),
                "task_type": task_context.task_type,
                "complexity": task_context.complexity.value,
                "agents_used": list(results.keys()),
                "success_rate": len([r for r in results.values() if r.get("success")]) / len(results) if results else 0,
                "final_success": final_response.get("success", False)
            }
            
            self.task_history.append(task_record)
            
            # Keep only last 1000 tasks
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-1000:]
            
            logger.info(f"Updated metrics for task: {task_context.task_type}")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the orchestrator."""
        try:
            if not self.task_history:
                return {"message": "No tasks completed yet"}
            
            total_tasks = len(self.task_history)
            successful_tasks = len([t for t in self.task_history if t["final_success"]])
            
            task_types = {}
            for task in self.task_history:
                task_type = task["task_type"]
                if task_type not in task_types:
                    task_types[task_type] = {"total": 0, "successful": 0}
                task_types[task_type]["total"] += 1
                if task["final_success"]:
                    task_types[task_type]["successful"] += 1
            
            return {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "overall_success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "task_type_breakdown": task_types,
                "recent_tasks": self.task_history[-10:] if self.task_history else []
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents."""
        try:
            health_results = {}
            
            for agent_id, agent in self.agents.items():
                try:
                    if hasattr(agent, 'health_check'):
                        health_result = await agent.health_check()
                    else:
                        # Basic health check
                        health_result = {"status": "healthy", "message": "Agent is running"}
                    
                    health_results[agent_id] = health_result
                except Exception as e:
                    health_results[agent_id] = {"status": "unhealthy", "error": str(e)}
            
            overall_status = "healthy" if all(
                result.get("status") == "healthy" 
                for result in health_results.values()
            ) else "degraded"
            
            return {
                "overall_status": overall_status,
                "agents": health_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {"error": str(e), "overall_status": "unhealthy"}

# Global orchestrator instance
orchestrator = AgentOrchestrator()
