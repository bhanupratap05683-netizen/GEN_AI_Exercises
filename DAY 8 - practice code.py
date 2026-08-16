# Complete CLI Agent: cli_agent/main.py

import asyncio
import json
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
import openai

console = Console()

class CLIDataAgent:
    def __init__(self):
        self.client = openai.AsyncOpenAI()
        self.conversation_history = []
        self.system_prompt = """You are a data assistant with access to:
1. SQL database (products, orders, customers)
2. REST APIs (weather, stocks, news)
3. Your own knowledge

Rules:
- Use tools when you need CURRENT or SPECIFIC data
- Respond directly for general questions
- Always explain what data you retrieved
- If a query fails, tell the user WHY and suggest alternatives"""
        
        self.tools = self._register_tools()
    
    def _register_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Execute SQL SELECT query on local database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql_query": {
                                "type": "string",
                                "description": "SQL SELECT statement"
                            }
                        },
                        "required": ["sql_query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_api",
                    "description": "Fetch data from a REST API endpoint",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Full URL to fetch"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "query_database":
            result = await safe_sql_query(args["sql_query"])
        elif tool_name == "fetch_api":
            result = await call_rest_api(args["url"])
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return json.dumps(result, default=str)
    
    async def chat(self, user_input: str) -> str:
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history
        
        max_iterations = 5
        
        for iteration in range(max_iterations):
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content
            
            # Show tool usage to user
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments)
                console.print(
                    f"[dim]🔧 Using tool: [bold]{tc.function.name}[/bold][/dim]"
                )
                if "sql_query" in args:
                    console.print(f"[dim]   SQL: {args['sql_query']}[/dim]")
            
            # Execute tools
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            for tc in message.tool_calls:
                result = await self.execute_tool(
                    tc.function.name,
                    json.loads(tc.function.arguments)
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
        
        return "I couldn't complete this in the allowed steps."
    
    async def run(self):
        console.print(Panel(
            "[bold blue]CLI Data Agent[/bold blue]\n"
            "Ask me to query databases, fetch APIs, or anything else!\n"
            "Type [bold]'exit'[/bold] to quit",
            title="🤖 AI Agent"
        ))
        
        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                if not user_input:
                    continue
                
                with console.status("[bold blue]Thinking...[/bold blue]"):
                    response = await self.chat(user_input)
                
                console.print(f"\n[bold blue]Agent:[/bold blue]")
                console.print(Markdown(response))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break

if __name__ == "__main__":
    agent = CLIDataAgent()
    asyncio.run(agent.run())
