# HANDS-ON: Durable Workflow Engine

import asyncio
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    step_id: str
    name: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    attempt: int = 0
    max_attempts: int = 3

@dataclass
class WorkflowState:
    workflow_id: str
    task: str
    steps: dict[str, WorkflowStep] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    status: str = "running"
    
    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "task": self.task,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "steps": {
                sid: {
                    "step_id": s.step_id,
                    "name": s.name,
                    "status": s.status.value,
                    "result": s.result,
                    "error": s.error,
                    "attempt": s.attempt
                }
                for sid, s in self.steps.items()
            }
        }

class DurableWorkflowEngine:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.cost_tracker: dict[str, float] = {}
    
    def _checkpoint_path(self, workflow_id: str) -> Path:
        return self.checkpoint_dir / f"{workflow_id}.json"
    
    def save_checkpoint(self, state: WorkflowState):
        path = self._checkpoint_path(state.workflow_id)
        with open(path, 'w') as f:
            json.dump(state.to_dict(), f, indent=2, default=str)
        print(f"💾 Checkpoint saved: {state.workflow_id}")
    
    def load_checkpoint(self, workflow_id: str) -> Optional[WorkflowState]:
        path = self._checkpoint_path(workflow_id)
        if not path.exists():
            return None
        
        with open(path) as f:
            data = json.load(f)
        
        state = WorkflowState(
            workflow_id=data["workflow_id"],
            task=data["task"],
            status=data["status"]
        )
        
        for sid, step_data in data["steps"].items():
            state.steps[sid] = WorkflowStep(
                step_id=step_data["step_id"],
                name=step_data["name"],
                status=StepStatus(step_data["status"]),
                result=step_data["result"],
                error=step_data["error"],
                attempt=step_data["attempt"]
            )
        
        print(f"📂 Checkpoint loaded: {workflow_id}")
        return state
    
    async def execute_step(
        self,
        state: WorkflowState,
        step_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        step_id = hashlib.md5(step_name.encode()).hexdigest()[:8]
        
        # Check if step already completed (idempotency)
        if step_id in state.steps:
            step = state.steps[step_id]
            if step.status == StepStatus.COMPLETED:
                print(f"⏭️  Step '{step_name}' already completed, skipping")
                return step.result
        
        # Create or get step
        if step_id not in state.steps:
            state.steps[step_id] = WorkflowStep(
                step_id=step_id,
                name=step_name
            )
        
        step = state.steps[step_id]
        
        while step.attempt < step.max_attempts:
            step.attempt += 1
            step.status = StepStatus.RUNNING
            step.started_at = time.time()
            self.save_checkpoint(state)
            
            try:
                print(f"▶️  Executing step: '{step_name}' (attempt {step.attempt})")
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                step.result = result
                step.status = StepStatus.COMPLETED
                step.completed_at = time.time()
                self.save_checkpoint(state)
                
                duration = step.completed_at - step.started_at
                print(f"✅ Step '{step_name}' completed in {duration:.2f}s")
                return result
                
            except Exception as e:
                step.error = str(e)
                print(f"❌ Step '{step_name}' failed (attempt {step.attempt}): {e}")
                
                if step.attempt < step.max_attempts:
                    wait_time = 2 ** step.attempt
                    print(f"   Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    step.status = StepStatus.FAILED
                    self.save_checkpoint(state)
                    raise Exception(f"Step '{step_name}' failed after {step.attempt} attempts: {e}")
    
    async def run_workflow(
        self,
        workflow_id: str,
        task: str,
        steps: list[tuple[str, Callable, list, dict]]
    ) -> WorkflowState:
        # Try to resume from checkpoint
        state = self.load_checkpoint(workflow_id)
        
        if not state:
            state = WorkflowState(
                workflow_id=workflow_id,
                task=task
            )
            print(f"🆕 Starting new workflow: {workflow_id}")
        else:
            print(f"🔄 Resuming workflow: {workflow_id}")
        
        try:
            for step_name, func, args, kwargs in steps:
                await self.execute_step(state, step_name, func, *args, **kwargs)
            
            state.status = "completed"
            state.completed_at = time.time()
            self.save_checkpoint(state)
            print(f"🎉 Workflow {workflow_id} completed!")
            
        except Exception as e:
            state.status = "failed"
            self.save_checkpoint(state)
            print(f"💥 Workflow {workflow_id} failed: {e}")
        
        return state

# Test durable workflow
async def main():
    engine = DurableWorkflowEngine()
    
    # Simulate steps with one failure
    call_count = 0
    
    async def fetch_data(source: str) -> dict:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.5)
        return {"data": f"fetched from {source}", "count": 100}
    
    async def process_data(data: dict) -> dict:
        await asyncio.sleep(0.3)
        return {"processed": True, "records": data["count"] * 2}
    
    async def save_results(results: dict) -> bool:
        await asyncio.sleep(0.2)
        return True
    
    steps = [
        ("fetch_external_data", fetch_data, ["https://api.example.com"], {}),
        ("process_records", process_data, [{"data": "sample", "count": 100}], {}),
        ("save_to_database", save_results, [{"processed": True, "records": 200}], {}),
    ]
    
    state = await engine.run_workflow(
        workflow_id="invoice-processing-001",
        task="Process monthly invoices",
        steps=steps
    )
    
    print(f"\nFinal status: {state.status}")
    for step_id, step in state.steps.items():
        print(f"  {step.name}: {step.status.value}")

asyncio.run(main())