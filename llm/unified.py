# llm/unified.py - Unified interface for all LLM backends
from __future__ import annotations

import os
from typing import Dict, Any, Optional

from llm import gemini, groq, custom

# Optional imports for OpenAI, Anthropic, and Hugging Face
try:
    from llm import openai
except ImportError:
    openai = None

try:
    from llm import anthropic
except ImportError:
    anthropic = None


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
        if openai is None:
            raise RuntimeError("OpenAI module not available. Install with: pip install openai")
        return _run_openai(paper_text, meta)
    elif "Claude" in model_choice:
        if anthropic is None:
            raise RuntimeError("Anthropic module not available. Install with: pip install anthropic")
        return _run_anthropic(paper_text, meta)
    elif "Llama" in model_choice or "Groq" in model_choice:
        return _run_groq(paper_text, meta)
    elif "Custom" in model_choice:
        return _run_custom(paper_text, meta)
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
    Uses shared utils.clean_and_ground (all LLMs use the same implementation).
    """
    from llm import utils
    return utils.clean_and_ground(
        raw, full_text,
        restrict_to_paper=restrict_to_paper,
        require_mutation_in_quote=require_mutation_in_quote,
        min_confidence=min_confidence
    )


# ========== Backend Implementations ==========

def _run_gemini(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Gemini backend - uses gemini.py which handles API key internally."""
    return gemini.run_on_paper(paper_text, meta)


def _run_groq(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Groq/Llama backend - uses groq.py which handles API key internally."""
    return groq.run_on_paper(paper_text, meta)


def _run_openai(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI GPT-4o backend - uses openai.py which handles API key internally."""
    return openai.run_on_paper(paper_text, meta)


def _run_anthropic(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic Claude backend - uses anthropic.py which handles API key internally."""
    return anthropic.run_on_paper(paper_text, meta)


def _run_custom(paper_text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Custom/backend-agnostic LLM endpoint."""
    return custom.run_on_paper(paper_text, meta)


# Helper functions are now in llm.utils - all LLMs use the same implementation


__all__ = ["run_on_paper", "clean_and_ground"]