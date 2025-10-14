from .gemini import run_on_paper as gemini_run_on_paper, clean_and_ground as gemini_clean_and_ground, DEFAULT_MODEL as GEMINI_DEFAULT_MODEL  # noqa: F401
try:
    from .groq import run_on_paper as groq_run_on_paper, clean_and_ground as groq_clean_and_ground, DEFAULT_MODEL as GROQ_DEFAULT_MODEL  # type: ignore  # noqa: F401
except Exception:
    # Groq may be optional; allow import to fail silently for environments without it
    pass

__all__ = [
    "gemini_run_on_paper",
    "gemini_clean_and_ground",
    "GEMINI_DEFAULT_MODEL",
    "groq_run_on_paper",
    "groq_clean_and_ground",
    "GROQ_DEFAULT_MODEL",
]


