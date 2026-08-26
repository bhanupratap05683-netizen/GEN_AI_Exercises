# HANDS-ON: DeepEval Test Suite

# pip install deepeval guardrails-ai

import asyncio
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric
)
import pytest

# Test cases as proper DeepEval objects
test_case_1 = LLMTestCase(
    input="What is the boiling point of water?",
    actual_output="Water boils at 100 degrees Celsius at sea level.",
    expected_output="100°C or 212°F",
    context=["Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm)."],
    retrieval_context=["Water boils at 100°C (212°F) at standard atmospheric pressure (1 atm)."]
)

# Define metrics
hallucination_metric = HallucinationMetric(threshold=0.5)
relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.7)

# pytest-style test
def test_rag_faithfulness():
    assert_test(test_case_1, [faithfulness_metric])

def test_rag_relevancy():
    assert_test(test_case_1, [relevancy_metric])

# Guardrails example
from guardrails import Guard
from guardrails.hub import ToxicLanguage, DetectPII, ValidJson

# Output validation guard
guard = Guard().use_many(
    ToxicLanguage(threshold=0.5, on_fail="exception"),
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix"),
)

async def safe_llm_call(prompt: str) -> str:
    import openai
    client = openai.AsyncOpenAI()
    
    response, *rest = guard(
        openai.chat.completions.create,
        prompt=prompt,
        model="gpt-4o-mini",
        max_tokens=256,
    )
    
    return response

# CI/CD GitHub Actions workflow (save as .github/workflows/evals.yml)
GITHUB_ACTIONS_WORKFLOW = """
name: LLM Eval Suite

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  run-evals:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install deepeval ragas pytest openai
    
    - name: Run evaluation suite
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python -m pytest tests/eval_suite.py -v --tb=short
    
    - name: Check pass rate
      run: |
        python scripts/check_metrics.py --min-pass-rate 0.85
"""