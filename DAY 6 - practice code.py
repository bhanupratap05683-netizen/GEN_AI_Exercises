# HANDS-ON: Complete Tool Calling System

import openai
import json
import asyncio
from typing import Any, Callable, get_type_hints
import inspect
from functools import wraps

# Tool registry
TOOLS: dict[str, dict] = {}
TOOL_FUNCTIONS: dict[str, Callable] = {}

def tool(description: str, **param_descriptions):
    """Decorator to register a function as an LLM tool"""
    def decorator(func: Callable) -> Callable:
        # Build JSON schema from function signature
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'return':
                continue
                
            python_type = hints.get(param_name, str)
            
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            
            properties[param_name] = {
                "type": type_map.get(python_type, "string"),
                "description": param_descriptions.get(
                    param_name, 
                    f"The {param_name} parameter"
                )
            }
            
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        tool_schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False
                }
            }
        }
        
        TOOLS[func.__name__] = tool_schema
        TOOL_FUNCTIONS[func.__name__] = func
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# Define actual tools
@tool(
    "Search for products in the database",
    query="Search keywords to find products",
    max_results="Maximum number of results to return (1-20)"
)
def search_products(query: str, max_results: int = 5) -> dict:
    # Simulated database
    products = [
        {"id": 1, "name": "Laptop Pro X", "price": 1299.99, "stock": 15},
        {"id": 2, "name": "Wireless Mouse", "price": 29.99, "stock": 150},
        {"id": 3, "name": "USB-C Hub", "price": 49.99, "stock": 75},
        {"id": 4, "name": "Monitor 4K", "price": 599.99, "stock": 8},
        {"id": 5, "name": "Keyboard Mechanical", "price": 149.99, "stock": 30},
    ]
    
    results = [
        p for p in products 
        if query.lower() in p["name"].lower()
    ][:max_results]
    
    return {"results": results, "total_found": len(results)}

@tool(
    "Get current weather for a city",
    city="City name to get weather for",
    units="Temperature units: celsius or fahrenheit"
)
def get_weather(city: str, units: str = "celsius") -> dict:
    # Simulated weather API
    weather_data = {
        "new york": {"temp": 22, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp": 15, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp": 28, "condition": "Sunny", "humidity": 70},
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower].copy()
        if units == "fahrenheit":
            data["temp"] = data["temp"] * 9/5 + 32
        data["units"] = units
        data["city"] = city
        return data
    
    return {"error": f"Weather data not available for {city}"}

@tool(
    "Execute a SQL query on the analytics database",
    query="SQL SELECT query to execute",
    database="Database name: analytics, sales, or inventory"
)
def execute_sql(query: str, database: str = "analytics") -> dict:
    # Security: only allow SELECT
    if not query.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are permitted"}
    
    # Simulate query results
    if "users" in query.lower():
        return {
            "columns": ["id", "name", "email", "created_at"],
            "rows": [
                [1, "Alice Johnson", "alice@example.com", "2024-01-15"],
                [2, "Bob Smith", "bob@example.com", "2024-01-16"],
            ],
            "row_count": 2
        }
    
    return {"columns": [], "rows": [], "row_count": 0}

# Tool execution engine
class ToolExecutor:
    def execute(self, tool_name: str, tool_args: dict) -> str:
        if tool_name not in TOOL_FUNCTIONS:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        func = TOOL_FUNCTIONS[tool_name]
        
        try:
            result = func(**tool_args)
            return json.dumps(result, default=str)
        except TypeError as e:
            return json.dumps({"error": f"Invalid arguments: {e}"})
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

# Complete agent loop with tool calling
async def tool_calling_agent(user_message: str) -> str:
    client = openai.AsyncOpenAI()
    executor = ToolExecutor()
    
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant with access to various tools.
Use tools when you need real data. Never make up data.
If a tool returns an error, explain it to the user."""
        },
        {"role": "user", "content": user_message}
    ]
    
    available_tools = list(TOOLS.values())
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # No tool calls = final answer
        if not message.tool_calls:
            return message.content
        
        # Add assistant message with tool calls
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
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"🔧 Calling tool: {tool_name}")
            print(f"   Args: {tool_args}")
            
            result = executor.execute(tool_name, tool_args)
            
            print(f"   Result: {result[:100]}...")
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return "Maximum iterations reached"

# Test it
async def main():
    questions = [
        "What's the weather like in London?",
        "Do you have any laptops in stock and what's the price?",
        "Search for wireless accessories under $100",
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"User: {question}")
        print(f"{'='*50}")
        answer = await tool_calling_agent(question)
        print(f"Agent: {answer}")

asyncio.run(main())