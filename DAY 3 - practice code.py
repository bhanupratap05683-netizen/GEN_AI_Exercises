# HANDS-ON: Complete Pydantic Structured Output System

from pydantic import BaseModel, Field, validator, model_validator
from typing import Optional, List, Literal
import openai
import json
from datetime import date

# Define complex nested schema
class LineItem(BaseModel):
    description: str = Field(..., description="What was purchased")
    quantity: float = Field(..., gt=0, description="Amount purchased")
    unit_price: float = Field(..., ge=0, description="Price per unit")
    total: float = Field(..., ge=0, description="quantity * unit_price")
    
    @model_validator(mode='after')
    def validate_total(self):
        expected = round(self.quantity * self.unit_price, 2)
        if abs(self.total - expected) > 0.01:
            raise ValueError(
                f"Total {self.total} doesn't match "
                f"quantity*price = {expected}"
            )
        return self

class Address(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "US"

class Invoice(BaseModel):
    invoice_number: str = Field(..., description="Invoice ID or number")
    invoice_date: Optional[str] = Field(None, description="Date in YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="Due date in YYYY-MM-DD")
    vendor_name: str = Field(..., description="Company issuing invoice")
    vendor_address: Optional[Address] = None
    customer_name: Optional[str] = None
    customer_address: Optional[Address] = None
    line_items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = Field(None, ge=0, le=1)
    tax_amount: Optional[float] = None
    total_amount: float = Field(..., description="Final total")
    currency: str = Field(default="USD")
    payment_terms: Optional[str] = None
    notes: Optional[str] = None
    confidence_score: float = Field(
        ..., ge=0, le=1,
        description="How confident are you in this extraction 0-1"
    )

# Retry-enabled extraction function
async def extract_with_retry(
    raw_text: str,
    schema: type[BaseModel],
    max_retries: int = 3
) -> BaseModel:
    client = openai.AsyncOpenAI()
    
    system_prompt = f"""You are an expert document parser.
Extract information from the provided document and return 
ONLY valid JSON matching this schema:
{json.dumps(schema.model_json_schema(), indent=2)}

Rules:
- Use null for missing fields
- Dates must be YYYY-MM-DD format
- Numbers must be numeric (not strings)
- confidence_score: 1.0 if all fields found, lower if guessing"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract from this document:\n\n{raw_text}"}
    ]
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0
            )
            
            raw_json = response.choices[0].message.content
            parsed = json.loads(raw_json)
            validated = schema.model_validate(parsed)
            
            print(f"✅ Extracted successfully on attempt {attempt + 1}")
            return validated
            
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
        except Exception as e:
            last_error = str(e)
        
        # Add error feedback for retry
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        messages.append({
            "role": "user",
            "content": f"""That response had an error: {last_error}
Please fix it and return valid JSON matching the schema exactly.
Common fixes:
- Ensure all required fields are present
- Check number formats (not strings)
- Verify date formats YYYY-MM-DD"""
        })
        
        print(f"⚠️  Attempt {attempt + 1} failed: {last_error}")
    
    raise Exception(f"Failed after {max_retries} attempts. Last error: {last_error}")

# Test it
messy_invoice = """
INVOICE
From: Acme Corporation, 123 Business St, NYC, NY 10001
To: Tech Startup Inc
Invoice #: INV-2024-0892
Date: January 15th, 2024  
Due: Feb 15, 2024

ITEMS:
- Cloud hosting services (3 months) .... 3 x $299.00 = $897.00
- Setup fee .................................. 1 x $150 = $150.00
- Support package .......................... 1 x $99.99 = $99.99

Subtotal: $1,146.99
Tax (8.5%): $97.49
TOTAL DUE: $1,244.48

Payment terms: Net 30
"""

import asyncio
result = asyncio.run(extract_with_retry(messy_invoice, Invoice))
print(result.model_dump_json(indent=2))