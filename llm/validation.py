# llm/validation.py - Agentic validation of extractions

# Validates all extractions by asking multiple questions per extraction
# Uses the same LLM backend as extraction
from __future__ import annotations

import json
import time
from typing import Dict, List, Any, Optional, Tuple

from llm import unified
from llm.prompts import PROMPTS
from llm.validation_questions import VALIDATION_QUESTIONS

def _build_validation_prompt(
    full_text: str,
    extraction: Dict[str, Any],
    questions: List[str],
    pmid: Optional[str] = None,
    pmcid: Optional[str] = None,
) -> str:
    """
    Build a validation prompt that asks multiple questions about an extraction.
    
    Args:
        full_text: Full text of the paper
        extraction: Single extraction/feature to validate
        questions: List of validation questions to ask
        pmid: Optional PMID for context
        pmcid: Optional PMCID for context
    
    Returns:
        Formatted prompt string
    """
    # Format extraction as reference
    extraction_str = json.dumps(extraction, indent=2, ensure_ascii=False)
    
    # Build questions section
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    # Build metadata
    meta_parts = []
    if pmid:
        meta_parts.append(f"PMID: {pmid}")
    if pmcid:
        meta_parts.append(f"PMCID: {pmcid}")
    meta_block = "\n".join(meta_parts) if meta_parts else ""
    
    # Prepare metadata prefix for full text (avoid backslash in f-string)
    newline_sep = "\n\n"
    meta_prefix = meta_block + newline_sep if meta_block else ""
    
    prompt = f"""You are a validation agent. Your task is to verify whether an extracted finding is accurate and present in the provided text.

EXTRACTION TO VALIDATE:
{extraction_str}

VALIDATION QUESTIONS:
{questions_text}

INSTRUCTIONS:
1. Read the full text carefully
2. Answer each question with "YES", "NO", or "UNCERTAIN"
3. Provide a brief rationale (1-2 sentences) for each answer
4. If the extraction is not found in the text, answer "NO" for all questions
5. If the extraction is found but details are incorrect, answer accordingly

RESPONSE FORMAT (JSON):
{{
  "answers": [
    {{"question": 1, "answer": "YES|NO|UNCERTAIN", "rationale": "brief explanation"}},
    {{"question": 2, "answer": "YES|NO|UNCERTAIN", "rationale": "brief explanation"}},
    {{"question": 3, "answer": "YES|NO|UNCERTAIN", "rationale": "brief explanation"}}
  ],
  "overall_validation": "PASS|FAIL|UNCERTAIN",
  "summary": "Overall assessment in 1-2 sentences"
}}

FULL TEXT:
{meta_prefix}{full_text}
"""
    return prompt

def _parse_validation_response(response: str, num_questions: int) -> Dict[str, Any]:
    """
    Parse LLM validation response.
    
    Returns:
        Dict with answers, overall_validation, summary, and parsed scores
    """
    # Try to extract JSON from response
    parsed = None
    
    # Strip code fences if present
    text = response.strip()
    if text.startswith("```"):
        text = text[3:]
        if "\n" in text:
            text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    
    # Try to find JSON object
    try:
        parsed = json.loads(text)
    except Exception:
        # Try to extract JSON object from text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end+1])
            except Exception:
                pass
    
    if not parsed or not isinstance(parsed, dict):
        # Fallback: return default structure
        return {
            "answers": [{"question": i+1, "answer": "UNCERTAIN", "rationale": "Failed to parse response"} 
                       for i in range(num_questions)],
            "overall_validation": "UNCERTAIN",
            "summary": "Failed to parse validation response",
            "scores": {"passed": 0, "total": num_questions, "accuracy": 0.0}
        }
    
    # Extract answers
    answers = parsed.get("answers", [])
    if not isinstance(answers, list):
        answers = []
    
    # Ensure we have answers for all questions
    while len(answers) < num_questions:
        answers.append({
            "question": len(answers) + 1,
            "answer": "UNCERTAIN",
            "rationale": "No answer provided"
        })
    
    # Score answers (YES = pass, NO/UNCERTAIN = fail)
    passed = sum(1 for a in answers if isinstance(a, dict) and 
                 a.get("answer", "").upper() == "YES")
    total = len(answers)
    accuracy = (passed / total * 100.0) if total > 0 else 0.0
    
    overall = parsed.get("overall_validation", "UNCERTAIN").upper()
    summary = parsed.get("summary", "No summary provided")
    
    return {
        "answers": answers[:num_questions],
        "overall_validation": overall,
        "summary": summary,
        "scores": {
            "passed": passed,
            "total": total,
            "accuracy": accuracy
        }
    }

def _generate_validation_questions(extraction: Dict[str, Any]) -> List[str]:
    """
    Generate 2-3 validation questions based on extraction type.
    
    Questions check:
    1. Presence: Is the feature/mutation mentioned?
    2. Accuracy: Are the positions/details correct?
    3. Context: Is the effect/function correctly attributed?
    
    Handles both bio schema format (with nested "feature") and legacy format.
    """
    questions = []
    
    # Handle both bio schema and legacy formats
    if "feature" in extraction and isinstance(extraction.get("feature"), dict):
        # Bio schema format
        feature = extraction.get("feature", {})
        mutation = extraction.get("mutation") or feature.get("name_or_label", "")
        protein = extraction.get("protein", "")
        virus = extraction.get("virus", "")
        position = extraction.get("position")
        effect = extraction.get("effect_or_function", {})
        effect_desc = effect.get("description", "") if isinstance(effect, dict) else ""
        residue_positions = feature.get("residue_positions", [])
    else:
        # Legacy format (already converted)
        feature = {}
        mutation = extraction.get("mutation", "")
        protein = extraction.get("protein", "")
        virus = extraction.get("virus", "")
        position = extraction.get("position")
        effect = {"description": extraction.get("effect_summary", "")}
        effect_desc = extraction.get("effect_summary", "")
        residue_positions = []
    
    # Get questions from template and format them with extraction data
    template_questions = VALIDATION_QUESTIONS.questions
    
    # Format each question with available placeholders
    for template_q in template_questions:
        try:
            # Format with available placeholders
            formatted_q = template_q.format(
                mutation=mutation or "",
                feature_name=feature.get("name_or_label", "") if isinstance(feature, dict) else "",
                protein=protein or "the",
                virus=virus or "the virus",
                position=str(position) if position is not None else "",
                residue_range=", ".join([f"{p.get('start')}-{p.get('end')}" 
                                       for p in (residue_positions or []) 
                                       if isinstance(p, dict) and p.get('start') is not None]) if residue_positions else "",
                effect_description=effect_desc[:150] if effect_desc else ""
            )
            questions.append(formatted_q)
        except (KeyError, ValueError):
            # If formatting fails (missing placeholder), use template as-is
            questions.append(template_q)
    
    # Ensure we have 1-5 questions
    if len(questions) == 0:
        # If no questions, add a default one
        questions.append("Is this extraction accurate and supported by the text?")
    elif len(questions) > 5:
        # Limit to 5 questions max
        questions = questions[:5]
    
    return questions

def validate_extraction(
    extraction: Dict[str, Any],
    full_text: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate a single extraction by asking multiple questions.
    
    Args:
        extraction: Single extraction/feature to validate
        full_text: Full text of the paper
        meta: LLM metadata (model_choice, model_name, api_key, etc.)
    
    Returns:
        Validation result with answers, scores, and overall assessment
    """
    meta = meta or {}
    
    # Generate validation questions
    questions = _generate_validation_questions(extraction)
    
    if not questions:
        return {
            "answers": [],
            "overall_validation": "UNCERTAIN",
            "summary": "Could not generate validation questions",
            "scores": {"passed": 0, "total": 0, "accuracy": 0.0}
        }
    
    # Build validation prompt
    prompt = _build_validation_prompt(
        full_text=full_text,
        extraction=extraction,
        questions=questions,
        pmid=meta.get("pmid"),
        pmcid=meta.get("pmcid"),
    )
    
    # Call LLM using unified interface (same backend as extraction)
    try:
        # Use the same LLM backend - route to appropriate completion function
        model_choice = meta.get("model_choice", "Gemini (Google)")
        
        # Route to appropriate backend's completion function
        if "Gemini" in model_choice:
            from llm import gemini
            import os
            # Ensure API key is set
            api_key = meta.get("api_key") or os.getenv("GEMINI_API_KEY")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            response = gemini._gemini_complete(prompt, max_output_tokens=2048)
        elif "GPT-4o" in model_choice:
            from llm import openai
            import os
            api_key = meta.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            model_name = meta.get("model_name", "gpt-4o-2024-11-20")
            response = openai._openai_complete(prompt, api_key, model_name, max_output_tokens=2048)
        elif "Claude" in model_choice:
            from llm import anthropic
            import os
            api_key = meta.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set")
            model_name = meta.get("model_name", "claude-sonnet-4-20250514")
            response = anthropic._anthropic_complete(prompt, api_key, model_name, max_output_tokens=2048)
        elif "Llama" in model_choice or "Groq" in model_choice:
            from llm import groq
            import os
            api_key = meta.get("api_key") or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set")
            model_name = meta.get("model_name", "llama-3.3-70b-versatile")
            response = groq._groq_complete(prompt, api_key, model_name, max_output_tokens=2048)
        elif "Custom" in model_choice:
            from llm import custom
            response = custom._custom_complete(
                prompt,
                api_url=meta.get("api_url"),
                api_key=meta.get("api_key"),
                extra_headers=meta.get("extra_headers"),
                model_name=meta.get("model_name"),
                timeout=meta.get("timeout", 120),
                openai_compatible=meta.get("openai_compatible", False),
                max_tokens=2048,
            )
        else:
            # Default to Gemini
            from llm import gemini
            import os
            api_key = meta.get("api_key") or os.getenv("GEMINI_API_KEY")
            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key
            response = gemini._gemini_complete(prompt, max_output_tokens=2048)
        
        # Parse response
        result = _parse_validation_response(response, len(questions))
        result["questions"] = questions
        
        # Add extraction row info for easy identification
        result["extraction_info"] = {
            "mutation": extraction.get("mutation") or "",
            "protein": extraction.get("protein") or "",
            "virus": extraction.get("virus") or "",
            "position": extraction.get("position"),
            "target_type": extraction.get("target_type") or "",
            "effect_summary": extraction.get("effect_summary") or "",
        }
        
        # Also include feature info if available
        if "feature" in extraction and isinstance(extraction.get("feature"), dict):
            feature = extraction.get("feature", {})
            result["extraction_info"]["feature_name"] = feature.get("name_or_label") or ""
            result["extraction_info"]["feature_type"] = feature.get("type") or ""
        
        return result
        
    except Exception as e:
        # Extract row info even on error
        extraction_info = {
            "mutation": extraction.get("mutation") or "",
            "protein": extraction.get("protein") or "",
            "virus": extraction.get("virus") or "",
            "position": extraction.get("position"),
            "target_type": extraction.get("target_type") or "",
            "effect_summary": extraction.get("effect_summary") or "",
        }
        if "feature" in extraction and isinstance(extraction.get("feature"), dict):
            feature = extraction.get("feature", {})
            extraction_info["feature_name"] = feature.get("name_or_label") or ""
            extraction_info["feature_type"] = feature.get("type") or ""
        
        return {
            "answers": [{"question": i+1, "answer": "UNCERTAIN", "rationale": f"Error: {str(e)}"} 
                       for i in range(len(questions))],
            "overall_validation": "UNCERTAIN",
            "summary": f"Validation failed: {str(e)}",
            "scores": {"passed": 0, "total": len(questions), "accuracy": 0.0},
            "questions": questions,
            "extraction_info": extraction_info,
            "error": str(e)
        }

def validate_all_extractions(
    extractions: List[Dict[str, Any]],
    full_text: str,
    meta: Optional[Dict[str, Any]] = None,
    accuracy_threshold: float = 85.0,
    delay_ms: int = 200,
) -> Dict[str, Any]:
    """
    Validate all extractions for a paper.
    
    Args:
        extractions: List of all extractions/features to validate
        full_text: Full text of the paper
        meta: LLM metadata (model_choice, model_name, api_key, etc.)
        accuracy_threshold: Minimum accuracy percentage to pass (default: 85.0)
        delay_ms: Delay between validation calls (default: 200ms)
    
    Returns:
        Validation summary with per-extraction results and overall accuracy
    """
    if not extractions:
        return {
            "total_extractions": 0,
            "total_validations": 0,
            "passed_validations": 0,
            "overall_accuracy": 0.0,
            "meets_threshold": False,
            "threshold": accuracy_threshold,
            "extraction_validations": [],
        }
    
    meta = meta or {}
    extraction_validations = []
    total_passed = 0
    total_validations = 0
    
    # Validate each extraction
    for idx, extraction in enumerate(extractions, 1):
        # Validate this extraction
        validation_result = validate_extraction(extraction, full_text, meta)
        
        # Extract scores
        scores = validation_result.get("scores", {})
        passed = scores.get("passed", 0)
        total = scores.get("total", 0)
        
        total_passed += passed
        total_validations += total
        
        # Store validation result with row info
        extraction_validations.append({
            "extraction_index": idx,
            "extraction": extraction,
            "validation": validation_result,
            # Add row identifier for easy lookup
            "row_id": f"extraction_{idx}",
            "mutation": extraction.get("mutation") or "",
            "protein": extraction.get("protein") or "",
            "virus": extraction.get("virus") or "",
            "position": extraction.get("position"),
        })
        
        # Delay between validations
        if delay_ms > 0 and idx < len(extractions):
            time.sleep(delay_ms / 1000.0)
    
    # Calculate overall accuracy
    overall_accuracy = (total_passed / total_validations * 100.0) if total_validations > 0 else 0.0
    meets_threshold = overall_accuracy >= accuracy_threshold
    
    return {
        "total_extractions": len(extractions),
        "total_validations": total_validations,
        "passed_validations": total_passed,
        "overall_accuracy": overall_accuracy,
        "meets_threshold": meets_threshold,
        "threshold": accuracy_threshold,
        "extraction_validations": extraction_validations,
    }

__all__ = [
    "validate_extraction",
    "validate_all_extractions",
    "_generate_validation_questions",
    "_build_validation_prompt",
    "_parse_validation_response",
]

