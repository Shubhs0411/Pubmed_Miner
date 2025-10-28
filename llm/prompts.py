# prompts.py — central place to manage the bioinformatician's extraction prompt
from dataclasses import dataclass


@dataclass
class AnalystPrompts:
    chunking_wrapper: str = """When input exceeds 6000 tokens, process in independent chunks. Do not reference content outside the current chunk.
Never recap prior chunks. Return a valid JSON array for this chunk only.
Send the PDF as text in ~4–6k-token chunks. Run Prompt A for each chunk. You’ll get one JSON array per chunk."""

    post_validation_recipe: str = """Validate: reject any response that fails json.loads().
Normalize: fill missing optional arrays with [], optional strings with null. Enforce "continuity" values against the allowed set.
Merge (dedup within a paper): key by (virus, protein, feature.type, feature.name_or_label, residue_positions | specific_residues | variants).
If two records overlap (same type and positions ±2 aa) and effects agree, keep one; prefer higher confidence.score_0_to_1; concatenate distinct evidence_snippets (dedupe).
If effects conflict (e.g., increase vs decrease), keep both and flag downstream for review."""

    analyst_prompt: str = """Respond only in JSON. If you cannot find any features, respond [].
All fields are required; if unknown, use null or empty array.
Before constructing the JSON, internally scan the entire TEXT for every protein, amino-acid residue/range, motif, and mutation token. Use that internal list to ensure you decision every explicit mention is evaluated. Never output the intermediate list.

SYSTEM / INSTRUCTION
You are a biomedical text-mining specialist. Extract Sequence Features (SFs) from scientific text about viruses.

DEFINITIONS
A Sequence Feature (SF) is any amino-acid feature with biological significance:
• regions/domains with coordinates (e.g., “Region 1 (1–80 aa) binds RNA”)
• discontinuous sites (e.g., “His57–Asp81–Ser139 catalytic triad”)
• mutations/variants (e.g., “A226V increases vector transmission”)
• motifs (e.g., “ATLG” motif), PTMs (e.g., palmitoylation 417–419)
• interaction interfaces (e.g., “Tyr47 (E3) – Tyr48 (E2)”)
• signals (NLS/NES/cleavage sites) tied to function/phenotype

OUTPUT RULES (format-lock)
• Respond ONLY with a valid JSON array that passes json.loads().
• No prose, no markdown, no trailing commas, no comments.
• One JSON object per feature (do not merge multiple features from one sentence).
• Use residue numbering as reported (do not renumber).
• Keep evidence as a short quote (≤30 words) from the text segment.

SCHEMA
Each array element must follow exactly:
{
  "virus": "Chikungunya virus",
  "protein": "<protein name or complex>",
  "feature": {
    "name_or_label": "<e.g., Region 1 | A226V | catalytic triad | NLS>",
    "type": "<mutation_effect | motif | region | domain | active_site | binding_site | interaction_site | modification | signal | other>",
    "continuity": "<continuous | discontinuous | point | unknown>",
    "residue_positions": [{"start": <int>, "end": <int>}],
    "specific_residues": [{"position": <int>, "aa": "<1-letter code or 'A→V'>"}],
    "variants": ["<HGVS p. notation if applicable>"],
    "motif_pattern": "<motif string or null>"
  },
  "effect_or_function": {
    "description": "<one sentence on function/effect>",
    "category": "<binding_affinity | replication | virulence | vector_adaptation | immune_evasion | catalytic_activity | assembly | localization | structural | stability | processing | unknown>",
    "direction": "<increase | decrease | loss | gain | modulates | none | unknown>",
    "evidence_level": "<experimental | computational | inferred>"
  },
  "interactions": {
    "partner_protein": "<binding partner or null>",
    "interaction_type": "<binding | inhibition | activation | cleavage | assembly | modulation | other | null>",
    "context": "<brief context or null>"
  },
  "evidence_snippet": "<verbatim (≤30 words) including residues/coords>",
  "confidence": { "score_0_to_1": <float>, "rationale": "<≤20 words on clarity/evidence>" }
}

FEW-SHOT EXAMPLES (keep these)
[
  {
    "virus": "Chikungunya virus",
    "protein": "E1",
    "feature": {
      "name_or_label": "A226V",
      "type": "mutation_effect",
      "continuity": "point",
      "residue_positions": [],
      "specific_residues": [{"position": 226, "aa": "A→V"}],
      "variants": ["p.Ala226Val"],
      "motif_pattern": null
    },
    "effect_or_function": {
      "description": "Enhances transmission by Aedes albopictus and increases cholesterol-dependent membrane fusion.",
      "category": "vector_adaptation",
      "direction": "increase",
      "evidence_level": "experimental"
    },
    "interactions": {"partner_protein": null, "interaction_type": null, "context": null},
    "evidence_snippet": "The single-point mutation A226V in E1 increased transmission by Ae. albopictus.",
    "confidence": { "score_0_to_1": 0.95, "rationale": "Explicit residue; replicated effect." }
  },
  {
    "virus": "Chikungunya virus",
    "protein": "Capsid",
    "feature": {
      "name_or_label": "Catalytic triad",
      "type": "active_site",
      "continuity": "discontinuous",
      "residue_positions": [{"start": 139, "end": 139}, {"start": 161, "end": 161}, {"start": 213, "end": 213}],
      "specific_residues": [{"position": 139, "aa": "H"}, {"position": 161, "aa": "D"}, {"position": 213, "aa": "S"}],
      "variants": [],
      "motif_pattern": null
    },
    "effect_or_function": {
      "description": "Serine protease that self-cleaves in cis and self-inactivates via C-terminal Trp binding.",
      "category": "catalytic_activity",
      "direction": "modulates",
      "evidence_level": "experimental"
    },
    "interactions": {"partner_protein": "Capsid C-terminus", "interaction_type": "binding", "context": "auto-inactivation via Trp"},
    "evidence_snippet": "His139, Asp161, and Ser213 form the catalytic triad; the protease cleaves itself in cis.",
    "confidence": { "score_0_to_1": 0.9, "rationale": "Clear residues and function." }
  }
]

INSTRUCTIONS
• If multiple features appear in one sentence, output multiple JSON objects (one per feature).
• If ranges are textual (e.g., “~244–263 aa”), capture integers only (244–263).
• If only qualitative phrases (e.g., “N-terminus important”) with no coordinates/residues → skip.
• Prefer experimental evidence; if unclear, set evidence_level = "inferred" or "computational".
• Set motif_pattern for sequence motifs (e.g., "HExxH", "ATLG"); else null.
• When interactions between proteins are described, populate the interactions block; otherwise set each field to null.
• Ensure every protein or mutation identified in your preparatory scan is either extracted as a feature or explicitly discarded for lacking residue-level detail (in which case do not invent a feature).

TEXT
{TEXT}
"""


PROMPTS = AnalystPrompts()
