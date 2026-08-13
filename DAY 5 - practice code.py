# schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class ExtractedInvoice(BaseModel):
    invoice_number: str
    vendor: str
    date: Optional[str]
    total: float
    line_items: List[dict]
    confidence: float

class ExtractionResult(BaseModel):
    success: bool
    document_path: str
    extracted_data: Optional[ExtractedInvoice]
    error: Optional[str]
    attempts: int
    processing_time_ms: int

# batch_runner.py
import asyncio
import time
from pathlib import Path
from typing import List

async def process_batch(
    document_paths: List[str],
    max_concurrent: int = 5
) -> List[ExtractionResult]:
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_one(path: str) -> ExtractionResult:
        start = time.time()
        async with semaphore:
            try:
                # Load and preprocess document
                text = await load_document(path)
                
                # Extract with retry
                data = await extract_with_retry(text, ExtractedInvoice)
                
                return ExtractionResult(
                    success=True,
                    document_path=path,
                    extracted_data=data,
                    error=None,
                    attempts=1,
                    processing_time_ms=int((time.time() - start) * 1000)
                )
            except Exception as e:
                return ExtractionResult(
                    success=False,
                    document_path=path,
                    extracted_data=None,
                    error=str(e),
                    attempts=3,
                    processing_time_ms=int((time.time() - start) * 1000)
                )
    
    results = await asyncio.gather(
        *[process_one(p) for p in document_paths]
    )
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"\n📊 Batch Results:")
    print(f"   Processed: {len(results)} documents")
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Failed: {len(results) - successful}")
    avg_time = sum(r.processing_time_ms for r in results) / len(results)
    print(f"   Avg time: {avg_time:.0f}ms per document")
    
    return results