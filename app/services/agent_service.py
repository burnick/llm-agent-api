"""Agent service implementation with LangChain integration."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.agent import BaseMultiActionAgent
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.models.interfaces import IAgentService, ILLMProvider, IToolService
from app.models.requests import AgentRequest
from app.models.responses import AgentResponse, AgentInfo, AgentCapability, ExecutionStep, TokenUsage
from app.models.context import ExecutionContext, Message, MessageRole
from app.models.errors import AgentExecutionError, ValidationError, ConfigurationError
from app.tools.registry import ToolRegistry
from app.tools.base import BaseTool as CustomBaseTool


logger = logging.getLogger(__name__)


class LangChainToolWrapper(BaseTool):
    """Wrapper to adapt our custom tools to LangChain's tool interface."""
    
    def __init__(self, custom_tool: CustomBaseTool, context: Optional[ExecutionContext] = None):
        self.custom_tool = custom_tool
        self.context = context
        super().__init__(
            name=custom_tool.name,
            description=custom_tool.description
        )
    
    def _run(self, **kwargs) -> str:
        """Synchronous run method (not used in async context)."""
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous run method."""
        try:
            from app.tools.context import ToolExecutionContext
            
            # Create tool execution context
            tool_context = ToolExecutionContext(
                session_id=self.context.session_id if self.context else str(uuid.uuid4()),
                user_id=self.context.user_id if self.context else None,
                user_permissions=["default"],  # TODO: Implement proper permissions
                max_execution_time=self.custom_tool.timeout
            )
            
            result = await self.custom_tool.execute(kwargs, tool_context)
            
            if result.status.value == "success":
                return str(result.result)
            else:
                return f"Tool execution failed: {result.error_message}"
                
        except Exception as e:
            logger.error(f"Error executing tool {self.custom_tool.name}: {e}")
            return f"Error: {str(e)}"


class AgentService(IAgentService):
    """Agent service implementation with LangChain integration."""
    
    def __init__(
        self, 
        llm_provider: ILLMProvider,
        tool_registry: ToolRegistry,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the agent service."""
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.config = config or {}
        
        # Agent configuration
        self.default_timeout = self.config.get("agent_timeout", 300.0)
        self.max_iterations = self.config.get("max_iterations", 10)
        self.memory_window_size = self.config.get("memory_window_size", 10)
        
        # Available agent types
        self.agent_types = {
            "default": {
                "name": "Default Agent",
                "description": "General purpose agent with access to all available tools",
                "system_prompt": "You are a helpful AI assistant. Use the available tools to help answer questions and complete tasks.",
                "capabilities": ["reasoning", "tool_usage", "conversation"]
            },
            "research": {
                "name": "Research Agent", 
                "description": "Specialized agent for research and information gathering",
                "system_prompt": "You are a research assistant. Focus on finding accurate information and providing well-sourced answers.",
                "capabilities": ["web_search", "information_synthesis", "fact_checking"]
            },
            "calculator": {
                "name": "Math Agent",
                "description": "Specialized agent for mathematical calculations and problem solving",
                "system_prompt": "You are a mathematics assistant. Help solve mathematical problems and perform calculations.",
                "capabilities": ["mathematical_reasoning", "calculations", "problem_solving"]
            }
        }
        
        logger.info(f"Initialized AgentService with {len(self.agent_types)} agent types")
    
    async def execute_agent(
        self, 
        request: AgentRequest, 
        context: Optional[ExecutionContext] = None
    ) -> AgentResponse:
        """Execute an agent with the given request and context."""
        start_time = time.time()
        execution_steps = []
        total_tokens_used = 0
        
        try:
            # Validate agent type
            if request.agent_type not in self.agent_types:
                raise ValidationError(f"Unknown agent type: {request.agent_type}")
            
            # Create or update execution context
            if context is None:
                context = await self.create_execution_context(
                    session_id=request.session_id or str(uuid.uuid4()),
                    user_id=None  # TODO: Extract from request or auth
                )
            
            # Add user message to context
            user_message = Message(
                role=MessageRole.USER,
                content=request.message,
                timestamp=datetime.utcnow()
            )
            context.add_message(user_message)
            
            # Log execution start
            init_step = ExecutionStep(
                step_type="initialization",
                description=f"Starting {request.agent_type} agent execution",
                input_data={
                    "message": request.message, 
                    "agent_type": request.agent_type,
                    "session_id": context.session_id,
                    "available_tools": request.tools
                },
                duration=0.0,
                timestamp=datetime.utcnow()
            )
            execution_steps.append(init_step)
            
            # Create LangChain agent
            agent_creation_start = time.time()
            agent_executor = await self._create_agent_executor(
                agent_type=request.agent_type,
                context=context,
                available_tools=request.tools
            )
            agent_creation_duration = time.time() - agent_creation_start
            
            execution_steps.append(ExecutionStep(
                step_type="agent_creation",
                description="Created LangChain agent executor with tools",
                input_data={"agent_type": request.agent_type},
                output_data={
                    "tools_count": len(agent_executor.tools),
                    "tool_names": [tool.name for tool in agent_executor.tools]
                },
                duration=agent_creation_duration,
                timestamp=datetime.utcnow()
            ))
            
            # Execute agent with detailed step tracking
            execution_result = await self._execute_with_detailed_tracking(
                agent_executor,
                request.message,
                context,
                execution_steps
            )
            
            # Process execution result
            response_content, intermediate_steps, tokens_used = await self._process_execution_result(
                execution_result,
                execution_steps
            )
            total_tokens_used += tokens_used
            
            # Add assistant message to context
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response_content,
                timestamp=datetime.utcnow(),
                metadata={
                    "agent_type": request.agent_type,
                    "tokens_used": tokens_used,
                    "intermediate_steps": len(intermediate_steps)
                }
            )
            context.add_message(assistant_message)
            
            # Final processing step
            execution_time = time.time() - start_time
            execution_steps.append(ExecutionStep(
                step_type="completion",
                description="Agent execution completed successfully",
                input_data={"total_steps": len(execution_steps)},
                output_data={
                    "response_length": len(response_content),
                    "total_tokens": total_tokens_used,
                    "execution_time": execution_time
                },
                duration=0.0,
                timestamp=datetime.utcnow()
            ))
            
            return AgentResponse(
                response=response_content,
                execution_steps=execution_steps,
                tokens_used=TokenUsage(
                    prompt_tokens=total_tokens_used // 2,  # Rough estimate
                    completion_tokens=total_tokens_used // 2,
                    total_tokens=total_tokens_used
                ),
                execution_time=execution_time,
                session_id=context.session_id,
                agent_type=request.agent_type
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Agent execution failed: {e}", exc_info=True)
            
            execution_steps.append(ExecutionStep(
                step_type="error",
                description=f"Agent execution failed: {str(e)}",
                input_data={"error": str(e), "error_type": type(e).__name__},
                output_data={"execution_time": execution_time},
                duration=execution_time,
                timestamp=datetime.utcnow()
            ))
            
            raise AgentExecutionError(
                f"Agent execution failed: {str(e)}",
                agent_type=request.agent_type,
                execution_steps=execution_steps
            ) from e
    
    async def _create_agent_executor(
        self,
        agent_type: str,
        context: ExecutionContext,
        available_tools: Optional[List[str]] = None
    ) -> AgentExecutor:
        """Create a LangChain agent executor."""
        try:
            # Get agent configuration
            agent_config = self.agent_types[agent_type]
            
            # Create LangChain-compatible LLM
            llm = await self._create_langchain_llm()
            
            # Get available tools
            tools = await self._get_langchain_tools(available_tools, context)
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", agent_config["system_prompt"]),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create memory
            memory = ConversationBufferWindowMemory(
                k=self.memory_window_size,
                memory_key="chat_history",
                return_messages=True
            )
            
            # Add conversation history to memory
            for message in context.get_recent_messages(self.memory_window_size):
                if message.role == MessageRole.USER:
                    memory.chat_memory.add_user_message(message.content)
                elif message.role == MessageRole.ASSISTANT:
                    memory.chat_memory.add_ai_message(message.content)
            
            # Create agent
            agent = create_openai_tools_agent(llm, tools, prompt)
            
            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                max_iterations=self.max_iterations,
                handle_parsing_errors=True
            )
            
            return agent_executor
            
        except Exception as e:
            logger.error(f"Failed to create agent executor: {e}")
            raise ConfigurationError(f"Failed to create agent executor: {str(e)}") from e
    
    async def _create_langchain_llm(self):
        """Create a LangChain-compatible LLM from our provider."""
        # For now, create a ChatOpenAI instance
        # TODO: Create a proper adapter for our ILLMProvider interface
        try:
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                timeout=30.0
            )
        except Exception as e:
            logger.error(f"Failed to create LangChain LLM: {e}")
            raise ConfigurationError(f"Failed to create LangChain LLM: {str(e)}") from e
    
    async def _get_langchain_tools(
        self, 
        available_tools: Optional[List[str]], 
        context: ExecutionContext
    ) -> List[BaseTool]:
        """Get LangChain-compatible tools."""
        tools = []
        
        # Get available tool names
        if available_tools:
            tool_names = [name for name in available_tools if name in self.tool_registry.list_tools()]
        else:
            tool_names = self.tool_registry.list_tools()
        
        # Wrap our custom tools for LangChain
        for tool_name in tool_names:
            custom_tool = self.tool_registry.get_tool(tool_name)
            if custom_tool:
                langchain_tool = LangChainToolWrapper(custom_tool, context)
                tools.append(langchain_tool)
        
        logger.info(f"Created {len(tools)} LangChain tools: {[t.name for t in tools]}")
        return tools
    
    async def _execute_with_timeout(
        self,
        agent_executor: AgentExecutor,
        query: str,
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute agent with timeout handling."""
        try:
            result = await asyncio.wait_for(
                agent_executor.ainvoke({"input": query}),
                timeout=self.default_timeout
            )
            return result
        except asyncio.TimeoutError:
            raise AgentExecutionError(
                f"Agent execution timed out after {self.default_timeout} seconds",
                agent_type="unknown",
                execution_steps=[]
            )
    
    async def _execute_with_detailed_tracking(
        self,
        agent_executor: AgentExecutor,
        query: str,
        context: ExecutionContext,
        execution_steps: List[ExecutionStep]
    ) -> Dict[str, Any]:
        """Execute agent with detailed step tracking."""
        execution_start = time.time()
        
        try:
            # Log the start of agent execution
            execution_steps.append(ExecutionStep(
                step_type="agent_start",
                description="Starting agent reasoning and tool execution",
                input_data={"query": query, "timeout": self.default_timeout},
                duration=0.0,
                timestamp=datetime.utcnow()
            ))
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent_executor.ainvoke({"input": query}),
                timeout=self.default_timeout
            )
            
            execution_duration = time.time() - execution_start
            
            # Log successful execution
            execution_steps.append(ExecutionStep(
                step_type="agent_complete",
                description="Agent execution completed successfully",
                input_data={"query": query},
                output_data={
                    "has_output": "output" in result,
                    "has_intermediate_steps": "intermediate_steps" in result,
                    "result_keys": list(result.keys())
                },
                duration=execution_duration,
                timestamp=datetime.utcnow()
            ))
            
            return result
            
        except asyncio.TimeoutError:
            execution_duration = time.time() - execution_start
            execution_steps.append(ExecutionStep(
                step_type="timeout",
                description=f"Agent execution timed out after {self.default_timeout} seconds",
                input_data={"timeout": self.default_timeout},
                duration=execution_duration,
                timestamp=datetime.utcnow()
            ))
            
            raise AgentExecutionError(
                f"Agent execution timed out after {self.default_timeout} seconds",
                agent_type="unknown",
                execution_steps=execution_steps
            )
        except Exception as e:
            execution_duration = time.time() - execution_start
            execution_steps.append(ExecutionStep(
                step_type="execution_error",
                description=f"Agent execution failed: {str(e)}",
                input_data={"error": str(e), "error_type": type(e).__name__},
                duration=execution_duration,
                timestamp=datetime.utcnow()
            ))
            raise
    
    async def _process_execution_result(
        self,
        result: Dict[str, Any],
        execution_steps: List[ExecutionStep]
    ) -> tuple[str, List[Dict[str, Any]], int]:
        """Process the execution result and extract detailed information."""
        processing_start = time.time()
        
        # Extract response content
        response_content = result.get("output", "")
        
        # Extract intermediate steps (tool calls, reasoning steps)
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Process intermediate steps for detailed tracking
        for i, step in enumerate(intermediate_steps):
            step_info = self._extract_step_info(step, i)
            execution_steps.append(ExecutionStep(
                step_type="intermediate_step",
                description=step_info["description"],
                input_data=step_info["input_data"],
                output_data=step_info["output_data"],
                duration=step_info.get("duration", 0.0),
                timestamp=datetime.utcnow()
            ))
        
        # Estimate token usage
        tokens_used = self._calculate_detailed_token_usage(result, intermediate_steps)
        
        processing_duration = time.time() - processing_start
        execution_steps.append(ExecutionStep(
            step_type="result_processing",
            description="Processed execution result and extracted information",
            input_data={"intermediate_steps_count": len(intermediate_steps)},
            output_data={
                "response_length": len(response_content),
                "tokens_estimated": tokens_used,
                "processing_duration": processing_duration
            },
            duration=processing_duration,
            timestamp=datetime.utcnow()
        ))
        
        return response_content, intermediate_steps, tokens_used
    
    def _extract_step_info(self, step: Any, step_index: int) -> Dict[str, Any]:
        """Extract information from an intermediate step."""
        try:
            # Handle different types of intermediate steps
            if hasattr(step, '__len__') and len(step) >= 2:
                action, observation = step[0], step[1]
                
                # Extract action information
                action_info = {
                    "tool": getattr(action, 'tool', 'unknown'),
                    "tool_input": getattr(action, 'tool_input', {}),
                    "log": getattr(action, 'log', '')
                }
                
                # Extract observation information
                observation_info = {
                    "result": str(observation) if observation else None,
                    "length": len(str(observation)) if observation else 0
                }
                
                return {
                    "description": f"Tool call {step_index + 1}: {action_info['tool']}",
                    "input_data": action_info,
                    "output_data": observation_info,
                    "duration": 0.0  # Duration not available from LangChain
                }
            else:
                return {
                    "description": f"Step {step_index + 1}: {type(step).__name__}",
                    "input_data": {"step_type": type(step).__name__},
                    "output_data": {"step_content": str(step)[:200]},
                    "duration": 0.0
                }
                
        except Exception as e:
            logger.warning(f"Failed to extract step info: {e}")
            return {
                "description": f"Step {step_index + 1}: Unable to parse",
                "input_data": {"error": str(e)},
                "output_data": {"raw_step": str(step)[:100]},
                "duration": 0.0
            }
    
    def _calculate_detailed_token_usage(
        self, 
        result: Dict[str, Any], 
        intermediate_steps: List[Any]
    ) -> int:
        """Calculate detailed token usage from execution result."""
        total_tokens = 0
        
        # Count tokens in main output
        output = result.get("output", "")
        total_tokens += self._estimate_token_usage(output, "")
        
        # Count tokens in intermediate steps
        for step in intermediate_steps:
            try:
                if hasattr(step, '__len__') and len(step) >= 2:
                    action, observation = step[0], step[1]
                    
                    # Count action tokens
                    action_text = getattr(action, 'log', '') + str(getattr(action, 'tool_input', {}))
                    total_tokens += len(action_text) // 4
                    
                    # Count observation tokens
                    observation_text = str(observation) if observation else ""
                    total_tokens += len(observation_text) // 4
                    
            except Exception as e:
                logger.warning(f"Failed to count tokens for step: {e}")
                # Add a small estimate for unparseable steps
                total_tokens += 10
        
        return total_tokens
    
    def _estimate_token_usage(self, input_text: str, output_text: str) -> int:
        """Estimate token usage for the interaction."""
        # Rough estimation: ~4 characters per token
        return (len(input_text) + len(output_text)) // 4
    
    async def list_available_agents(self) -> List[str]:
        """Get a list of available agent types."""
        return list(self.agent_types.keys())
    
    async def get_agent_info(self, agent_type: str) -> AgentInfo:
        """Get detailed information about a specific agent type."""
        if agent_type not in self.agent_types:
            raise ValidationError(f"Unknown agent type: {agent_type}")
        
        config = self.agent_types[agent_type]
        
        # Convert capabilities to AgentCapability objects
        capabilities = [
            AgentCapability(
                name=cap,
                description=f"Agent supports {cap}",
                parameters={}
            )
            for cap in config["capabilities"]
        ]
        
        return AgentInfo(
            agent_type=agent_type,
            name=config["name"],
            description=config["description"],
            capabilities=capabilities,
            available_tools=self.tool_registry.list_tools()
        )
    
    async def get_agent_capabilities(self, agent_type: str) -> Dict[str, Any]:
        """Get the capabilities of a specific agent type."""
        if agent_type not in self.agent_types:
            raise ValidationError(f"Unknown agent type: {agent_type}")
        
        config = self.agent_types[agent_type]
        
        # Get detailed tool information
        available_tools = self.tool_registry.list_tools()
        tool_definitions = {}
        for tool_name in available_tools:
            tool_def = self.tool_registry.get_tool_definition(tool_name)
            if tool_def:
                tool_definitions[tool_name] = {
                    "description": tool_def.description,
                    "parameters": tool_def.parameters,
                    "required_permissions": tool_def.required_permissions,
                    "timeout": tool_def.timeout
                }
        
        # Get LLM provider capabilities
        llm_capabilities = await self._get_llm_capabilities()
        
        return {
            "agent_type": agent_type,
            "name": config["name"],
            "description": config["description"],
            "capabilities": config["capabilities"],
            "system_prompt": config["system_prompt"],
            "available_tools": available_tools,
            "tool_definitions": tool_definitions,
            "execution_limits": {
                "max_iterations": self.max_iterations,
                "timeout": self.default_timeout,
                "memory_window": self.memory_window_size
            },
            "llm_capabilities": llm_capabilities,
            "supported_features": [
                "conversation_memory",
                "tool_calling",
                "step_tracking",
                "timeout_handling",
                "error_recovery"
            ]
        }
    
    async def _get_llm_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of the underlying LLM provider."""
        try:
            # Get basic provider info
            provider_info = {
                "provider_name": self.llm_provider.provider_name,
                "available_models": self.llm_provider.available_models
            }
            
            # Test connection
            is_connected = await self.llm_provider.validate_connection()
            provider_info["connection_status"] = "connected" if is_connected else "disconnected"
            
            return provider_info
            
        except Exception as e:
            logger.warning(f"Failed to get LLM capabilities: {e}")
            return {
                "provider_name": "unknown",
                "available_models": [],
                "connection_status": "error",
                "error": str(e)
            }
    
    async def discover_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Discover and return capabilities for all available agent types."""
        capabilities = {}
        
        for agent_type in await self.list_available_agents():
            try:
                capabilities[agent_type] = await self.get_agent_capabilities(agent_type)
            except Exception as e:
                logger.error(f"Failed to get capabilities for agent {agent_type}: {e}")
                capabilities[agent_type] = {
                    "error": str(e),
                    "status": "unavailable"
                }
        
        return capabilities
    
    async def get_execution_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent service usage and performance."""
        # This would typically be implemented with proper metrics collection
        # For now, return basic configuration information
        return {
            "service_info": {
                "available_agent_types": len(self.agent_types),
                "available_tools": len(self.tool_registry.list_tools()),
                "default_timeout": self.default_timeout,
                "max_iterations": self.max_iterations,
                "memory_window_size": self.memory_window_size
            },
            "agent_types": list(self.agent_types.keys()),
            "tool_registry_stats": self.tool_registry.get_registry_stats(),
            "configuration": {
                "timeout": self.default_timeout,
                "max_iterations": self.max_iterations,
                "memory_window": self.memory_window_size
            }
        }
    
    async def create_execution_context(
        self, 
        session_id: str, 
        user_id: Optional[str] = None
    ) -> ExecutionContext:
        """Create a new execution context for an agent session."""
        return ExecutionContext(
            session_id=session_id,
            user_id=user_id,
            conversation_history=[],
            available_tools=self.tool_registry.list_tools(),
            execution_metadata={
                "created_by": "agent_service",
                "version": "1.0.0"
            }
        )
    
    async def update_execution_context(self, context: ExecutionContext) -> ExecutionContext:
        """Update an existing execution context."""
        context.updated_at = datetime.utcnow()
        return context