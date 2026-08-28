# HANDS-ON: Custom Tracing System

import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Optional, Any
from contextlib import asynccontextmanager
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Simple in-memory tracer (no external service needed)
@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "OK"
    error: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def finish(self, error: Optional[str] = None):
        self.end_time = time.time()
        if error:
            self.status = "ERROR"
            self.error = error

class TraceCollector:
    def __init__(self):
        self.spans: list[Span] = []
        self.current_trace_id: Optional[str] = None
        self.current_span_stack: list[str] = []
    
    def start_trace(self, name: str) -> Span:
        import uuid
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:8]
        
        self.current_trace_id = trace_id
        
        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            start_time=time.time()
        )
        
        self.spans.append(span)
        self.current_span_stack = [span_id]
        return span
    
    def start_span(self, name: str) -> Span:
        import uuid
        span_id = str(uuid.uuid4())[:8]
        parent_id = self.current_span_stack[-1] if self.current_span_stack else None
        
        span = Span(
            name=name,
            trace_id=self.current_trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            start_time=time.time()
        )
        
        self.spans.append(span)
        self.current_span_stack.append(span_id)
        return span
    
    def end_span(self, span: Span, error: Optional[str] = None):
        span.finish(error)
        if span.span_id in self.current_span_stack:
            self.current_span_stack.remove(span.span_id)
    
    def print_trace(self, trace_id: str = None):
        tid = trace_id or self.current_trace_id
        trace_spans = [s for s in self.spans if s.trace_id == tid]
        
        print(f"\n{'='*70}")
        print(f"TRACE: {tid}")
        print(f"{'='*70}")
        
        for span in trace_spans:
            depth = 0
            parent = span.parent_span_id
            while parent:
                parent_span = next((s for s in trace_spans if s.span_id == parent), None)
                if parent_span:
                    depth += 1
                    parent = parent_span.parent_span_id
                else:
                    break
            
            indent = "  " * depth
            status_icon = "✅" if span.status == "OK" else "❌"
            duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "..."
            
            print(f"{indent}{status_icon} [{span.span_id}] {span.name} ({duration})")
            
            if span.attributes:
                for k, v in span.attributes.items():
                    val_str = str(v)[:60]
                    print(f"{indent}   📎 {k}: {val_str}")
            
            if span.error:
                print(f"{indent}   ❌ Error: {span.error[:80]}")

# Global tracer
tracer = TraceCollector()

@asynccontextmanager
async def traced_llm_call(
    operation_name: str,
    model: str,
    prompt_tokens_estimate: int = 0
):
    span = tracer.start_span(f"llm.{operation_name}")
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.prompt_tokens_estimate", prompt_tokens_estimate)
    
    try:
        yield span
        tracer.end_span(span)
    except Exception as e:
        tracer.end_span(span, error=str(e))
        raise

@asynccontextmanager
async def traced_tool_call(tool_name: str, args: dict):
    span = tracer.start_span(f"tool.{tool_name}")
    span.set_attribute("tool.name", tool_name)
    span.set_attribute("tool.args", json.dumps(args)[:200])
    
    try:
        yield span
        tracer.end_span(span)
    except Exception as e:
        tracer.end_span(span, error=str(e))
        raise

# Instrumented agent
async def traced_agent_run(task: str):
    root_span = tracer.start_trace("agent.run")
    root_span.set_attribute("agent.task", task[:200])
    
    import openai
    client = openai.AsyncOpenAI()
    
    # Step 1: Planning
    async with traced_llm_call("planning", "gpt-4o") as span:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Plan how to: {task}"}]
        )
        plan = response.choices[0].message.content
        span.set_attribute("llm.output_tokens", response.usage.completion_tokens)
        span.set_attribute("llm.input_tokens", response.usage.prompt_tokens)
    
    # Step 2: Tool call
    async with traced_tool_call("web_search", {"query": task}) as span:
        await asyncio.sleep(0.1)  # Simulated tool call
        result = f"Search results for: {task}"
        span.set_attribute("tool.result_length", len(result))
    
    # Step 3: Synthesis
    async with traced_llm_call("synthesis", "gpt-4o-mini") as span:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Based on: {result}\nAnswer: {task}"}
            ]
        )
        final = response.choices[0].message.content
        span.set_attribute("llm.output_tokens", response.usage.completion_tokens)
    
    root_span.finish()
    tracer.print_trace()
    
    return final

asyncio.run(traced_agent_run("What is the latest news about AI?"))