# Day Learning: Structured Outputs, Pydantic V2 & LLM Retry Loops

A comprehensive summary and reference guide on guaranteeing reliable, typed, and schema-compliant outputs from Large Language Models (LLMs) using **Pydantic V2**, **OpenAI's Structured Output modes**, and **Error-Feedback Retry Loops**.

---

## 📑 Table of Contents
1. [Overview & Core Motivation](#overview--core-motivation)
2. [Why Structured Outputs?](#why-structured-outputs)
3. [Pydantic V2 Essentials](#pydantic-v2-essentials)
4. [OpenAI Structured Output Modes](#openai-structured-output-modes)
5. [Self-Correction & Retry Loops](#self-correction--retry-loops)
6. [Code Reference & Architecture](#code-reference--architecture)
7. [Key Takeaways & Production Guidelines](#key-takeaways--production-guidelines)

---

## 💡 Overview & Core Motivation

In production AI systems, relying on unconstrained LLM text generation introduces non-determinism, fragile parsing logic, and dynamic runtime errors. Moving from prompt engineering to **guaranteed schema enforcement** is a fundamental requirement for building robust LLM pipelines.

This module covers the core theoretical and practical frameworks necessary to transition LLMs from raw text engines into typed, deterministic backend components.

---

## 🎯 Why Structured Outputs?

LLMs natively operate on probabilistic string generation—everything returned by a base model is text. 

* **The Raw Prompting Bottleneck:** Instructing a model via prompts like `"Return valid JSON with keys X, Y, Z"` frequently fails in production. Models may output markdown blocks (```json), trailing commas, missing required keys, or incorrect data types.
* **Production Reliability Standard:** Software architectures depend on strong typing and contract enforcement. Unvalidated inputs break downstream APIs, database insertions, and business logic.
* **Guaranteed Schema Compliance:** Structured outputs bridge natural language processing with standard software engineering by guaranteeing that the returned payload conforms exactly to a expected type schema before downstream code executes.

---

## 🛡️ Pydantic V2 Essentials

[Pydantic V2](https://docs.pydantic.dev/) is the industry standard data validation and settings management library in Python. It relies on native type hints to enforce dynamic data validation at runtime.

### Core Concepts & Components

| Component | Description & Usage |
| :--- | :--- |
| `BaseModel` | The foundational class used to define custom data schemas using standard Python type annotations. |
| `Field()` | Provides field-level metadata, default values, numerical/string range constraints (`ge`, `le`, `min_length`), and human-readable descriptions for LLM tool selection. |
| `@field_validator` | Custom validation methods to enforce complex validation rules (e.g., specific string formatting, range validation across fields). |
| `model_validate()` | Parses and validates raw Python dictionaries or data structures into a validated class instance. |
| `model_json_schema()` | Automatically generates OpenAPI/JSON Schema compliance specs, enabling standard integration with LLM APIs. |
| `ValidationError` | Raised whenever parsing fails. Provides structured debug details explicitly identifying which field failed and why. |

---

## ⚡ OpenAI Structured Output Modes

OpenAI provides three primary patterns for retrieving structured data from models, each suited to different architecture requirements.

### 1. Mode 1: `response_format={"type": "json_object"}`
* **Mechanism:** Forces the model to generate syntactically valid JSON (averts malformed syntax errors).
* **Limitation:** Does **not** enforce key structure, field types, or presence of required attributes. System prompts must manually define and request specific keys.

### 2. Mode 2: Strict Schema Enforcement (`beta.chat.completions.parse` / Pydantic)
* **Mechanism:** OpenAI accepts a Pydantic model / strict JSON Schema and uses constrained decoding techniques at the sampling level.
* **Advantage:** Guarantees 100% strict adherence to the schema. Output is parsed directly into a Pydantic object without structural failure.

### 3. Mode 3: Tool Calling / Function Calling
* **Mechanism:** Defines structured outputs as functions/tools available to the LLM (`tools=[...]`).
* **Advantage:** High flexibility. Supports multi-tool selection, execution routing, and dynamic argument extraction while maintaining strict parameter schemas.

---

## 🔁 Self-Correction & Retry Loops

Despite strict schemas, validation can fail due to domain logic violations (e.g., invalid business rules validated by custom `@field_validator` logic). A self-healing retry loop handles these runtime edge cases automatically.

```
       ┌────────────────────────┐
       │   Prompt LLM Model     │
       └───────────┬────────────┘
                   │
                   ▼
       ┌────────────────────────┐
       │ Validate with Pydantic │
       └───────────┬────────────┘
                   │
         ┌─────────┴─────────┐
         │ Is Schema Valid?  │
         └────┬─────────┬────┘
           Pass         Fail
          │             │
          ▼             ▼
  ┌─────────────┐  ┌─────────────────────────────────────┐
  │ Return Data │  │ Append Error Feedback to History     │
  └─────────────┘  │ (Include failed response + error msg)│
                   └──────────────────┬──────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ Retry Generation    │
                           │ (Max Retries: 3-5)  │
                           └─────────────────────┘
```

### Self-Correction Strategy
1. **Catch Errors:** Capture exact tracebacks or custom error descriptions from `ValidationError`.
2. **Feedback Context:** Feed the prior invalid LLM output along with the specific validation error message back into the conversation history.
3. **Explicit Directive:** Prompt the model to correct only the broken fields in its next iteration.
4. **Retry Threshold:** Maintain a maximum retry count (typically 3 to 5 attempts). If retries are exhausted, raise a controlled exception to protect system resources.

---

## 💻 Reference Implementation

Below is a complete Python pattern combining Pydantic V2 validation and a self-healing retry loop:

```python
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError

# 1. Define schema using Pydantic V2
class UserProfile(BaseModel):
    username: str = Field(..., description="Unique alphanumeric username")
    age: int = Field(..., ge=18, le=120, description="Age must be between 18 and 120")
    skills: List[str] = Field(default_factory=list, description="List of primary technical skills")

    @field_validator('username')
    @classmethod
    def validate_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError('Username must be strictly alphanumeric')
        return v

# 2. Resilient Parsing with Retry Loop
def parse_llm_response_with_retry(raw_llm_callable, initial_prompt: str, max_retries: int = 3) -> UserProfile:
    prompt = initial_prompt
    history = [{"role": "user", "content": prompt}]
    
    for attempt in range(1, max_retries + 1):
        try:
            # Execute LLM call
            raw_response = raw_llm_callable(history)
            
            # Parse & validate response using Pydantic model
            profile = UserProfile.model_validate_json(raw_response)
            return profile
            
        except (ValidationError, ValueError) as e:
            print(f"[Attempt {attempt}/{max_retries}] Validation failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(f"Failed to generate valid schema after {max_retries} attempts.") from e
            
            # Construct feedback prompt for self-correction
            feedback = f"Your previous response failed validation with the following error:
{e}
Please correct the errors and return only valid data."
            history.append({"role": "assistant", "content": raw_response})
            history.append({"role": "user", "content": feedback})
```

---

## 📌 Key Takeaways & Production Guidelines

1. **Never parse LLM output with regex or raw `json.loads` without a validator schema.** Always wrap incoming JSON in a Pydantic schema or type-checking boundary.
2. **Use Descriptions in `Field()`:** Descriptions serve a dual purpose—they document code and act as high-precision instructions for the model's schema generation.
3. **Leverage Native Structured Outputs when available:** Prefer OpenAI's strict `response_format` or tool calling capabilities to minimize retry loop overhead.
4. **Always set `max_retries`:** Infinite loops in LLM workflows cause runaway costs and high latencies. Bound all self-correction loops to 3–5 iterations.
README.md
Displaying README.md.