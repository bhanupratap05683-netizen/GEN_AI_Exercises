# HANDS-ON: State Machine Agent

import asyncio
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import time
import json

class AgentState(Enum):
    IDLE = auto()
    PLANNING = auto()
    EXECUTING_TOOL = auto()
    EVALUATING = auto()
    WAITING_HUMAN = auto()
    SELF_CORRECTING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class TrajectoryStep:
    step_number: int
    state: AgentState
    thought: str
    action: Optional[str]
    action_args: Optional[dict]
    observation: Optional[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "step": self.step_number,
            "state": self.state.name,
            "thought": self.thought,
            "action": self.action,
            "args": self.action_args,
            "observation": self.observation[:200] if self.observation else None
        }

@dataclass
class AgentContext:
    task: str
    state: AgentState = AgentState.IDLE
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    error_count: int = 0
    max_errors: int = 3
    max_steps: int = 15
    human_approvals: list[dict] = field(default_factory=list)
    result: Optional[str] = None
    
    def add_step(self, step: TrajectoryStep):
        self.trajectory.append(step)
        print(f"\n[Step {step.step_number}] State: {step.state.name}")
        print(f"  Thought: {step.thought[:100]}...")
        if step.action:
            print(f"  Action: {step.action}({step.action_args})")
    
    def transition(self, new_state: AgentState):
        old_state = self.state
        self.state = new_state
        print(f"  → Transition: {old_state.name} → {new_state.name}")

class ReActStateMachineAgent:
    VALID_TRANSITIONS = {
        AgentState.IDLE: [AgentState.PLANNING],
        AgentState.PLANNING: [AgentState.EXECUTING_TOOL, AgentState.WAITING_HUMAN, AgentState.COMPLETED],
        AgentState.EXECUTING_TOOL: [AgentState.EVALUATING, AgentState.SELF_CORRECTING],
        AgentState.EVALUATING: [AgentState.PLANNING, AgentState.COMPLETED, AgentState.FAILED],
        AgentState.WAITING_HUMAN: [AgentState.PLANNING, AgentState.FAILED],
        AgentState.SELF_CORRECTING: [AgentState.EXECUTING_TOOL, AgentState.FAILED],
        AgentState.COMPLETED: [],
        AgentState.FAILED: [],
    }
    
    # Sensitive actions that require human approval
    SENSITIVE_ACTIONS = {"send_email", "delete_record", "make_payment", "deploy_code"}
    
    def __init__(self, tools: dict[str, Callable]):
        self.tools = tools
        self.client = __import__('openai').AsyncOpenAI()
    
    def _validate_transition(self, ctx: AgentContext, new_state: AgentState):
        valid = self.VALID_TRANSITIONS.get(ctx.state, [])
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition: {ctx.state.name} → {new_state.name}. "
                f"Valid: {[s.name for s in valid]}"
            )
    
    async def _llm_step(self, ctx: AgentContext) -> dict:
        """Get next action from LLM"""
        
        trajectory_text = "\n".join([
            f"Step {s.step_number}: {s.thought} → {s.action or 'NO_ACTION'}"
            for s in ctx.trajectory[-5:]  # Last 5 steps
        ])
        
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])
        
        system = f"""You are an agent that solves tasks step by step.

Available tools:
{tool_descriptions}

Rules:
- Think step by step
- Use tools to gather information
- Return JSON with your decision
- If task is complete, set action to "FINISH"
- If you need human approval, set action to "REQUEST_HUMAN"

Response format:
{{
    "thought": "your reasoning",
    "action": "tool_name or FINISH or REQUEST_HUMAN",
    "args": {{"param": "value"}},
    "requires_human": false,
    "is_complete": false,
    "final_answer": null
}}"""
        
        recent_trajectory = trajectory_text if trajectory_text else "No steps yet"
        errors = f"\nErrors so far: {ctx.error_count}" if ctx.error_count > 0 else ""
        
        user = f"""Task: {ctx.task}

Recent steps:
{recent_trajectory}{errors}

What should I do next?"""
        
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def _request_human_approval(
        self, 
        ctx: AgentContext,
        action: str, 
        args: dict
    ) -> bool:
        """Simulate human approval (in production: send notification, wait)"""
        print(f"\n⚠️  HUMAN APPROVAL REQUIRED")
        print(f"   Action: {action}")
        print(f"   Arguments: {json.dumps(args, indent=4)}")
        
        # In production: send to queue, wait for response
        # For now: simulate auto-approval for safe actions
        response = input("   Approve? (y/n): ").strip().lower()
        approved = response == 'y'
        
        ctx.human_approvals.append({
            "action": action,
            "args": args,
            "approved": approved,
            "timestamp": time.time()
        })
        
        return approved
    
    async def run(self, task: str) -> dict:
        ctx = AgentContext(task=task)
        step_number = 0
        
        ctx.transition(AgentState.PLANNING)
        
        while ctx.state not in [AgentState.COMPLETED, AgentState.FAILED]:
            step_number += 1
            
            if step_number > ctx.max_steps:
                ctx.transition(AgentState.FAILED)
                ctx.result = "Exceeded maximum steps"
                break
            
            # Get LLM decision
            try:
                decision = await self._llm_step(ctx)
            except Exception as e:
                ctx.error_count += 1
                if ctx.error_count >= ctx.max_errors:
                    ctx.transition(AgentState.FAILED)
                    ctx.result = f"LLM failure: {e}"
                    break
                continue
            
            thought = decision.get("thought", "")
            action = decision.get("action", "FINISH")
            args = decision.get("args", {})
            
            # Handle completion
            if decision.get("is_complete") or action == "FINISH":
                step = TrajectoryStep(
                    step_number=step_number,
                    state=AgentState.COMPLETED,
                    thought=thought,
                    action="FINISH",
                    action_args=None,
                    observation=None
                )
                ctx.add_step(step)
                
                self._validate_transition(ctx, AgentState.COMPLETED)
                ctx.transition(AgentState.COMPLETED)
                ctx.result = decision.get("final_answer", "Task completed")
                break
            
            # Check if human approval needed
            if action in self.SENSITIVE_ACTIONS or decision.get("requires_human"):
                self._validate_transition(ctx, AgentState.WAITING_HUMAN)
                ctx.transition(AgentState.WAITING_HUMAN)
                
                approved = await self._request_human_approval(ctx, action, args)
                
                if not approved:
                    step = TrajectoryStep(
                        step_number=step_number,
                        state=AgentState.WAITING_HUMAN,
                        thought=thought,
                        action=action,
                        action_args=args,
                        observation="Human rejected this action"
                    )
                    ctx.add_step(step)
                    
                    self._validate_transition(ctx, AgentState.FAILED)
                    ctx.transition(AgentState.FAILED)
                    ctx.result = "Action rejected by human"
                    break
                
                self._validate_transition(ctx, AgentState.PLANNING)
                ctx.transition(AgentState.PLANNING)
            
            # Execute tool
            if action in self.tools:
                self._validate_transition(ctx, AgentState.EXECUTING_TOOL)
                ctx.transition(AgentState.EXECUTING_TOOL)
                
                try:
                    tool_func = self.tools[action]
                    observation = await tool_func(**args) if asyncio.iscoroutinefunction(tool_func) \
                        else tool_func(**args)
                    observation = str(observation)
                    
                    step = TrajectoryStep(
                        step_number=step_number,
                        state=AgentState.EXECUTING_TOOL,
                        thought=thought,
                        action=action,
                        action_args=args,
                        observation=observation
                    )
                    ctx.add_step(step)
                    
                    self._validate_transition(ctx, AgentState.EVALUATING)
                    ctx.transition(AgentState.EVALUATING)
                    self._validate_transition(ctx, AgentState.PLANNING)
                    ctx.transition(AgentState.PLANNING)
                    
                except Exception as e:
                    ctx.error_count += 1
                    
                    step = TrajectoryStep(
                        step_number=step_number,
                        state=AgentState.EXECUTING_TOOL,
                        thought=thought,
                        action=action,
                        action_args=args,
                        observation=f"ERROR: {e}"
                    )
                    ctx.add_step(step)
                    
                    if ctx.error_count >= ctx.max_errors:
                        ctx.transition(AgentState.FAILED)
                        ctx.result = f"Too many errors: {e}"
                        break
                    
                    self._validate_transition(ctx, AgentState.SELF_CORRECTING)
                    ctx.transition(AgentState.SELF_CORRECTING)
                    self._validate_transition(ctx, AgentState.EXECUTING_TOOL)
                    ctx.transition(AgentState.EXECUTING_TOOL)
        
        return {
            "task": task,
            "result": ctx.result,
            "final_state": ctx.state.name,
            "steps_taken": step_number,
            "errors": ctx.error_count,
            "human_approvals": ctx.human_approvals,
            "trajectory": [s.to_dict() for s in ctx.trajectory]
        }

# Test the agent
async def main():
    def get_weather(city: str) -> str:
        """Get current weather for a city"""
        return f"Weather in {city}: 22°C, partly cloudy"
    
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression safely"""
        try:
            allowed = set('0123456789+-*/()., ')
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return "Invalid expression"
        except:
            return "Calculation error"
    
    def search_web(query: str) -> str:
        """Search the web for information"""
        return f"Search results for '{query}': Found 3 relevant articles about {query}"
    
    tools = {
        "get_weather": get_weather,
        "calculate": calculate,
        "search_web": search_web
    }
    
    agent = ReActStateMachineAgent(tools)
    
    result = await agent.run(
        "What's the weather in London and Tokyo? Also calculate the average of 22 and 28."
    )
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {result['result']}")
    print(f"State: {result['final_state']}")
    print(f"Steps: {result['steps_taken']}")

asyncio.run(main())