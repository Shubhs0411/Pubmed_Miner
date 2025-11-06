# llm/unified.py - Unified interface for all LLM backends
from __future__ import annotations

import os
from typing import Dict, Any, Optional

from llm import gemini, groq


def run_on_paper(paper_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Unified entry point for all LLM backends.
    Routes to appropriate backend based on meta['model_choice'].
    
    Args:
        paper_text: Full text of the paper
        meta: Contains model_choice, model_name, api_key, chunk_chars, etc.
    
    Returns:
        Dict with extracted features
    """
    meta = meta or {}
    model_choice = meta.get("model_choice", "Gemini (Google)")
    
    # Route to appropriate backend
    if "Gemini" in model_choice:
        return _run_gemini(paper_text, meta)
    elif "GPT-4o" in model_choice:
        return _run_openai(paper_text, meta)
    elif "Claude" in model_choice:
        return _run_anthropic(paper_text, meta)
    elif "Llama" in model_choice or "Groq" in model_choice:
        return _run_groq(paper_text, meta)
    else:
        # Default to Gemini
        return _run_gemini(paper_text, meta)


def clean_and_ground(raw: Dict[str, Any],
                     full_text: str,
                     *,
                     restrict_to_paper: bool = True,
                     require_mutation_in_quote: bool = False,
                     min_confidence: float = 0.0) -> Dict[str, Any]:
    """
    Unified cleaning/validation for all backends.
    Currently uses Gemini's implementation as it's most comprehensive.
    """
    return gemini.clean_and_ground(
        raw, full_text,
        restrict_to_paper=restrict_to_paper,
        require_mutation_in_quote=require_mutation_in_quote,
        min_confidence=min_confidence
    )


# ========== Backend Implementations ==========

def _run_gemini(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Gemini backend (existing implementation)"""
    # Set API key if provided
    api_key = meta.get("api_key")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    # Set model name
    model_name = meta.get("model_name", "gemini-2.0-flash-exp")
    os.environ["GEMINI_MODEL"] = model_name
    
    # Reload gemini module with new settings
    import importlib
    importlib.reload(gemini)
    
    return gemini.run_on_paper(paper_text, meta)


def _run_groq(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Groq/Llama backend (existing implementation)"""
    api_key = meta.get("api_key")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    # Note: Groq doesn't support custom model selection in current impl
    # You may need to modify groq.py to accept model_name parameter
    
    return groq.run_on_paper(paper_text, meta)


def _run_openai(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenAI GPT-4o backend.
    
    You'll need to:
    1. pip install openai
    2. Create this implementation similar to gemini.py
    """
    import openai
    import json
    import time
    from llm.prompts import PROMPTS
    
    api_key = meta.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    client = openai.OpenAI(api_key=api_key)
    model_name = meta.get("model_name", "gpt-4o-2024-11-20")
    
    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")
    chunk_chars = int(meta.get("chunk_chars") or 16000)
    overlap_chars = int(meta.get("overlap_chars") or 500)
    delay_ms = int(meta.get("delay_ms") or 0)
    
    # Chunk text
    chunks = _chunk_text(paper_text, max_chars=chunk_chars, overlap=overlap_chars)
    
    all_features = []
    
    for idx, chunk in enumerate(chunks[:4], 1):  # Limit to 4 chunks
        # Build prompt
        prompt = PROMPTS.analyst_prompt.replace("{TEXT}", chunk)
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a biomedical text extraction expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=8192,
            )
            
            raw_text = response.choices[0].message.content
            
            # Parse JSON
            parsed = _safe_json_value(raw_text)
            
            if isinstance(parsed, dict) and "sequence_features" in parsed:
                features = parsed["sequence_features"]
            elif isinstance(parsed, list):
                features = parsed
            else:
                features = []
            
            all_features.extend(features)
            
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
                
        except Exception as e:
            print(f"[OpenAI] Error on chunk {idx}: {e}")
            continue
    
    return {
        "paper": {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": meta.get("title"),
            "virus_candidates": [],
            "protein_candidates": []
        },
        "sequence_features": all_features
    }


def _run_anthropic(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Anthropic Claude backend.
    
    You'll need to:
    1. pip install anthropic
    2. Create this implementation similar to gemini.py
    """
    import anthropic
    import json
    import time
    from llm.prompts import PROMPTS
    
    api_key = meta.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    model_name = meta.get("model_name", "claude-sonnet-4-20250514")
    
    pmid = meta.get("pmid")
    pmcid = meta.get("pmcid")
    chunk_chars = int(meta.get("chunk_chars") or 16000)
    overlap_chars = int(meta.get("overlap_chars") or 500)
    delay_ms = int(meta.get("delay_ms") or 0)
    
    # Chunk text
    chunks = _chunk_text(paper_text, max_chars=chunk_chars, overlap=overlap_chars)
    
    all_features = []
    
    for idx, chunk in enumerate(chunks[:4], 1):  # Limit to 4 chunks
        # Build prompt
        prompt = PROMPTS.analyst_prompt.replace("{TEXT}", chunk)
        
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=8192,
                temperature=0.0,
                system="You are a biomedical text extraction expert. Respond only with valid JSON.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            raw_text = response.content[0].text
            
            # Parse JSON
            parsed = _safe_json_value(raw_text)
            
            if isinstance(parsed, dict) and "sequence_features" in parsed:
                features = parsed["sequence_features"]
            elif isinstance(parsed, list):
                features = parsed
            else:
                features = []
            
            all_features.extend(features)
            
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
                
        except Exception as e:
            print(f"[Claude] Error on chunk {idx}: {e}")
            continue
    
    return {
        "paper": {
            "pmid": pmid,
            "pmcid": pmcid,
            "title": meta.get("title"),
            "virus_candidates": [],
            "protein_candidates": []
        },
        "sequence_features": all_features
    }


# ========== Helper Functions ==========

def _chunk_text(text: str, max_chars: int = 12000, overlap: int = 500):
    """Split text into overlapping chunks"""
    text = " ".join(text.split())  # normalize whitespace
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    
    return chunks


def _safe_json_value(raw: str):
    """Parse JSON with error handling"""
    import json
    
    if not isinstance(raw, str):
        return None
    
    s = raw.strip()
    
    # Strip code fences
    if s.startswith("```"):
        s = s[3:]
        if "\n" in s:
            s = s.split("\n", 1)[1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    
    # Try extracting array
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    
    # Try extracting object
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    
    return None


__all__ = ["run_on_paper", "clean_and_ground"]