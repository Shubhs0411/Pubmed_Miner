# llm/validation_questions.py - Validation question templates
# Separate from prompts.py for better organization

from typing import List


class ValidationQuestionTemplates:
    """
    Simple validation question list.
    Users can edit 1-5 questions through the UI.
    
    Placeholders you can use in questions:
    - {mutation}: The mutation name (e.g., "A226V")
    - {feature_name}: The feature name (e.g., "RNA-binding domain")
    - {protein}: The protein name (e.g., "E1", "nsP3")
    - {virus}: The virus name (e.g., "Chikungunya virus")
    - {position}: Single position (e.g., "226")
    - {residue_range}: Residue range (e.g., "1-73")
    - {effect_description}: Effect/function description
    """
    
    def __init__(self):
        # Default questions (1-5 questions, one per line)
        self._questions: List[str] = [
            "Is there experimental evidence for a sequence feature in {protein} protein of {virus}?",
            "Are the residue positions and structural boundaries correctly identified for this feature in {protein}, with supporting experimental evidence?",
            "Is the biological function or role of this feature in {protein} correctly described, and is this supported by experimental evidence (binding assays, enzymatic activity, structural studies, etc.)?"
        ]
    
    @property
    def questions(self) -> List[str]:
        """Get the list of validation questions."""
        return self._questions.copy()
    
    @questions.setter
    def questions(self, value: List[str]):
        """Set the list of validation questions."""
        if isinstance(value, list):
            # Filter out empty questions and limit to 5
            self._questions = [q.strip() for q in value if q.strip()][:5]
        elif isinstance(value, str):
            # If string provided, split by newlines
            self._questions = [q.strip() for q in value.split('\n') if q.strip()][:5]
    
    def get_default_questions(self) -> List[str]:
        """Get default questions."""
        return [
            "Is there experimental evidence for a sequence feature in {protein} protein of {virus}?",
            "Are the residue positions and structural boundaries correctly identified for this feature in {protein}, with supporting experimental evidence?",
            "Is the biological function or role of this feature in {protein} correctly described, and is this supported by experimental evidence (binding assays, enzymatic activity, structural studies, etc.)?"
        ]
    
    def reset_to_default(self):
        """Reset questions to defaults."""
        self._questions = self.get_default_questions()


# Global validation question templates instance
VALIDATION_QUESTIONS = ValidationQuestionTemplates()

