# Import all LLM modules
from . import gemini, utils  # noqa: F401

# Optional imports
try:
    from . import groq  # noqa: F401
except Exception:
    groq = None

try:
    from . import openai  # noqa: F401
except Exception:
    openai = None

try:
    from . import anthropic  # noqa: F401
except Exception:
    anthropic = None

try:
    from . import custom  # noqa: F401
except Exception:
    custom = None

# Re-export for backward compatibility
from .gemini import run_on_paper as gemini_run_on_paper, clean_and_ground as gemini_clean_and_ground, DEFAULT_MODEL as GEMINI_DEFAULT_MODEL  # noqa: F401

try:
    from .groq import run_on_paper as groq_run_on_paper, clean_and_ground as groq_clean_and_ground, DEFAULT_MODEL as GROQ_DEFAULT_MODEL  # type: ignore  # noqa: F401
except Exception:
    groq_run_on_paper = None
    groq_clean_and_ground = None
    GROQ_DEFAULT_MODEL = None

try:
    from .openai import run_on_paper as openai_run_on_paper, clean_and_ground as openai_clean_and_ground, DEFAULT_MODEL as OPENAI_DEFAULT_MODEL  # noqa: F401
except Exception:
    openai_run_on_paper = None
    openai_clean_and_ground = None
    OPENAI_DEFAULT_MODEL = None

try:
    from .anthropic import run_on_paper as anthropic_run_on_paper, clean_and_ground as anthropic_clean_and_ground, DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL  # noqa: F401
except Exception:
    anthropic_run_on_paper = None
    anthropic_clean_and_ground = None
    ANTHROPIC_DEFAULT_MODEL = None

try:
    from .custom import run_on_paper as custom_run_on_paper, clean_and_ground as custom_clean_and_ground, DEFAULT_MODEL as CUSTOM_DEFAULT_MODEL, DEFAULT_URL as CUSTOM_DEFAULT_URL  # noqa: F401
except Exception:
    custom_run_on_paper = None
    custom_clean_and_ground = None
    CUSTOM_DEFAULT_MODEL = None
    CUSTOM_DEFAULT_URL = None

__all__ = [
    # Modules (available if imports succeeded)
    "gemini",
    "utils",
    # Functions
    "gemini_run_on_paper",
    "gemini_clean_and_ground",
    "GEMINI_DEFAULT_MODEL",
    "groq_run_on_paper",
    "groq_clean_and_ground",
    "GROQ_DEFAULT_MODEL",
    "openai_run_on_paper",
    "openai_clean_and_ground",
    "OPENAI_DEFAULT_MODEL",
    "anthropic_run_on_paper",
    "anthropic_clean_and_ground",
    "ANTHROPIC_DEFAULT_MODEL",
    "custom_run_on_paper",
    "custom_clean_and_ground",
    "CUSTOM_DEFAULT_MODEL",
    "CUSTOM_DEFAULT_URL",
]


