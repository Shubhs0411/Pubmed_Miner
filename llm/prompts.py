# prompts.py — UPDATED to extract structural domains and regions
from dataclasses import dataclass


@dataclass
class AnalystPrompts:
    chunking_wrapper: str = """When input exceeds 6000 tokens, process in independent chunks. Do not reference content outside the current chunk.
Never recap prior chunks. Return a valid JSON array for this chunk only.
Send the PDF as text in ~4–6k-token chunks. Run Prompt A for each chunk. You'll get one JSON array per chunk."""

    post_validation_recipe: str = """Validate: reject any response that fails json.loads().
Normalize: fill missing optional arrays with [], optional strings with null. Enforce "continuity" values against the allowed set.
Merge (dedup within a paper): key by (virus, protein, feature.type, feature.name_or_label, residue_positions | specific_residues | variants).
If two records overlap (same type and positions ±2 aa) and effects agree, keep one; prefer higher confidence.score_0_to_1; concatenate distinct evidence_snippets (dedupe).
If effects conflict (e.g., increase vs decrease), keep both and flag downstream for review."""

    # Locked sections (cannot be edited - critical for app functionality)
    _prompt_header: str = """Respond only in JSON. If you cannot find any features, respond [].
All fields are required; if unknown, use null or empty array.
Before constructing the JSON, internally scan the entire TEXT for every protein, amino-acid residue/range, motif, mutation token, AND structural domain/region. Use that internal list to ensure every explicit mention is evaluated. Never output the intermediate list.
"""

    _prompt_output_rules: str = """OUTPUT RULES (format-lock)
• Respond ONLY with a valid JSON array that passes json.loads().
• No prose, no markdown, no trailing commas, no comments.
• One JSON object per feature (regions can be split if they have different functions).
• Use residue numbering as reported (do not renumber).
• Keep evidence as a short quote (≤30 words) from the text segment.
• **For structural regions**: set type="region" or "domain", populate residue_positions with the range(s).
"""

    _prompt_schema: str = """SCHEMA
Each array element must follow exactly:
{
  "virus": "Chikungunya virus",
  "protein": "<protein name or complex>",
  "feature": {
    "name_or_label": "<e.g., RNA-binding domain | Region 1 | A226V | catalytic triad | NLS | linker region | C-terminal tail>",
    "type": "<mutation_effect | motif | region | domain | active_site | binding_site | interaction_site | modification | signal | linker | disordered_region | other>",
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
"""

    _prompt_examples: str = """FEW-SHOT EXAMPLES (keep these)
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
  },
  {
    "virus": "Chikungunya virus",
    "protein": "nsP3",
    "feature": {
      "name_or_label": "RNA-binding domain",
      "type": "domain",
      "continuity": "continuous",
      "residue_positions": [{"start": 1, "end": 73}],
      "specific_residues": [],
      "variants": [],
      "motif_pattern": null
    },
    "effect_or_function": {
      "description": "Double-stranded RNA-binding domain responsible for viral RNA recognition.",
      "category": "binding_affinity",
      "direction": "none",
      "evidence_level": "experimental"
    },
    "interactions": {"partner_protein": "viral RNA", "interaction_type": "binding", "context": "dsRNA recognition"},
    "evidence_snippet": "The first 73 amino acids constitute the double-stranded RNA-binding domain.",
    "confidence": { "score_0_to_1": 0.85, "rationale": "Clear boundaries and function." }
  },
  {
    "virus": "Chikungunya virus",
    "protein": "nsP3",
    "feature": {
      "name_or_label": "Effector domain",
      "type": "domain",
      "continuity": "continuous",
      "residue_positions": [{"start": 85, "end": 207}],
      "specific_residues": [],
      "variants": [],
      "motif_pattern": null
    },
    "effect_or_function": {
      "description": "Effector domain (ED) responsible for downstream signaling and interactions.",
      "category": "structural",
      "direction": "none",
      "evidence_level": "experimental"
    },
    "interactions": {"partner_protein": null, "interaction_type": null, "context": null},
    "evidence_snippet": "the last 85–207 amino acids constitute the ED.",
    "confidence": { "score_0_to_1": 0.85, "rationale": "Clear boundaries and structural role." }
  },
  {
    "virus": "Chikungunya virus",
    "protein": "nsP3",
    "feature": {
      "name_or_label": "Flexible linker region",
      "type": "linker",
      "continuity": "continuous",
      "residue_positions": [{"start": 74, "end": 84}],
      "specific_residues": [],
      "variants": [],
      "motif_pattern": null
    },
    "effect_or_function": {
      "description": "Flexible linker connecting RNA-binding and effector domains.",
      "category": "structural",
      "direction": "none",
      "evidence_level": "inferred"
    },
    "interactions": {"partner_protein": null, "interaction_type": null, "context": null},
    "evidence_snippet": "The two domains are connected by a flexible linker region (LR).",
    "confidence": { "score_0_to_1": 0.75, "rationale": "Clear structural role but approximate range." }
  },
  {
    "virus": "Chikungunya virus",
    "protein": "nsP3",
    "feature": {
      "name_or_label": "C-terminal disordered tail",
      "type": "disordered_region",
      "continuity": "continuous",
      "residue_positions": [{"start": 178, "end": 207}],
      "specific_residues": [],
      "variants": [],
      "motif_pattern": null
    },
    "effect_or_function": {
      "description": "Intrinsically disordered C-terminal tail, likely involved in protein-protein interactions.",
      "category": "structural",
      "direction": "none",
      "evidence_level": "inferred"
    },
    "interactions": {"partner_protein": null, "interaction_type": null, "context": null},
    "evidence_snippet": "The last 30 residues form the C-terminal disordered tail.",
    "confidence": { "score_0_to_1": 0.8, "rationale": "Clear structural annotation with coordinates." }
  }
]
"""

    _prompt_footer: str = """TEXT
{TEXT}
"""

    # Editable sections (can be modified by users)
    # Part 1: SYSTEM/INSTRUCTION and DEFINITIONS (comes before OUTPUT RULES)
    analyst_prompt_editable_part1: str = """SYSTEM / INSTRUCTION
You are a biomedical text-mining specialist. Extract Sequence Features (SFs) from scientific text about viruses.

DEFINITIONS
A Sequence Feature (SF) is any amino-acid feature with biological significance:
• **STRUCTURAL REGIONS/DOMAINS** with coordinates (e.g., "amino acids 1-73 form the RNA-binding domain", "residues 85-207 constitute the effector domain", "C-terminal tail (178-207 aa)")
• **FUNCTIONAL DOMAINS** (e.g., "catalytic domain 50-150", "transmembrane region 200-220")
• **LINKER REGIONS** (e.g., "flexible linker 74-84 aa")
• **DISORDERED REGIONS** (e.g., "disordered tail residues 178-207")
• discontinuous sites (e.g., "His57–Asp81–Ser139 catalytic triad")
• **mutations/variants** (e.g., "A226V increases vector transmission")
• motifs (e.g., "ATLG" motif), PTMs (e.g., palmitoylation 417-419)
• interaction interfaces (e.g., "Tyr47 (E3) – Tyr48 (E2)")
• signals (NLS/NES/cleavage sites) tied to function/phenotype

**CRITICAL**: Extract ALL structural regions even if they have NO mutation or variant. A domain description with coordinates IS a sequence feature.

PATTERN RECOGNITION GUIDE
Systematically scan the text for these patterns. Extract ALL instances you find:

**Mutation Patterns:**
• Standard format: Single letter + number + single letter (e.g., A226V, K128E, S123A, D456F)
• With asterisk: Single letter + number + asterisk (e.g., W123*, Q456*)
• Reversed format: Number + single letter (e.g., 226V, 128E) - convert to standard format
• HGVS protein notation: p.SingleThreeDigitSingleThree (e.g., p.Ala226Val, p.Lys128Glu, p.Ser123Ala)
• Spelled mutations: "alanine 226 to valine", "lysine 128 to glutamic acid", "residue 226 changed from A to V"
• Substitution phrases: "A226V substitution", "mutation at position 226 (A→V)", "226A/V"

**Protein Patterns:**
• Protein names followed by "protein": "E1 protein", "NS3 protein", "capsid protein", "envelope protein"
• Protein names with descriptors: "E1 glycoprotein", "NS3 polyprotein", "E2 envelope protein"
• Protein complexes: "E1-E2 complex", "capsid-subunit", "protein complex"
• Abbreviations: Look for protein abbreviations (E1, E2, NS1, NS2, NS3, NS4, NS5, C, M, etc.)

**Residue Number Patterns:**
• Single residues: "residue 226", "position 128", "amino acid 456", "aa 123"
• Ranges: "residues 1-73", "amino acids 420-440", "residues 85 to 207", "1-460 aa"
• Range phrases: "residues 1 through 73", "from residue 85 to 207", "spanning residues 420-440"
• Count references: "73 amino acids", "460 aa", "207 residues" (may indicate domain boundaries)

**Amino Acid Position Patterns:**
• Single letter + position: "A226", "K128", "S123", "D456"
• Three-letter + position: "Ala226", "Lys128", "Ser123", "Asp456"
• Full name + position: "alanine 226", "lysine 128", "serine 123", "aspartic acid 456"
• With context: "residue A226", "position K128", "amino acid Ser123"

**Structural Domain Patterns:**
• Domain with coordinates: "RNA-binding domain (residues 1-73)", "effector domain (85-207 aa)", "catalytic domain spanning residues 50-150"
• Domain phrases: "amino acids 1-73 form the RNA-binding domain", "residues 85-207 constitute the effector domain"
• Transmembrane: "transmembrane domain at residues 420-440", "TM domain (420-440)", "transmembrane region 200-220"
• Linker regions: "flexible linker 74-84 aa", "linker region (74-84)", "connecting residues 74-84"
• Terminal regions: "C-terminal tail (178-207 aa)", "N-terminal domain (1-100)", "C-terminus (residues 450-460)"

**Motif Patterns:**
• Sequence motifs: "HExxH motif", "ATLG motif", "GxGxxG motif"
• Pattern descriptions: "conserved sequence ATLG", "motif pattern HExxH", "signature sequence"
• Short sequences: 3-10 uppercase letters, possibly with x/X for variable positions

**Coverage Strategy:**
1. First pass: Systematically scan the ENTIRE text for ALL patterns listed above
2. For each pattern found, extract it as a feature with full context
3. Do not skip any pattern - if you see it, extract it
4. Pay special attention to mutations in any format (standard, HGVS, spelled, etc.)
5. Extract structural domains even if they don't have mutations
6. Ensure no mutation or domain is missed due to format variations
"""

    # Part 2: INSTRUCTIONS (comes after EXAMPLES)
    analyst_prompt_editable_part2: str = """INSTRUCTIONS
• **CRITICAL**: Extract ALL structural domains, regions, linkers, and tails WITH their coordinates, even if no mutation is mentioned.
• **PATTERN COVERAGE**: Use the PATTERN RECOGNITION GUIDE above to systematically find ALL mutations, proteins, residues, domains, and motifs. Do not miss any pattern variant.
• **MUTATION FORMATS**: Extract mutations in ALL formats - standard (A226V), HGVS (p.Ala226Val), spelled ("alanine 226 to valine"), and substitution phrases. Convert all to standard format when possible.
• **COMPREHENSIVE SCANNING**: Before extracting, internally list ALL mutations, proteins, residues, and domains you find. Ensure every one is either extracted or explicitly noted as lacking detail.
• If multiple features appear in one sentence, output multiple JSON objects (one per feature).
• If ranges are textual (e.g., "~244–263 aa"), capture integers only (244–263).
• For regions spanning the sentence (e.g., "1-73 aa"), calculate the range if "last X residues" is mentioned.
• If only qualitative phrases (e.g., "N-terminus important") with no coordinates → skip.
• Prefer experimental evidence; if unclear, set evidence_level = "inferred" or "computational".
• Set motif_pattern for sequence motifs (e.g., "HExxH", "ATLG"); else null.
• When interactions between proteins are described, populate the interactions block; otherwise set each field to null.
• **NO MISSING PATTERNS**: If you see a mutation pattern (in any format), a protein mention, a residue number, or a domain description, you MUST extract it. Do not skip patterns due to format variations.
"""
    
    # Combined editable section for UI (combines part1 and part2)
    @property
    def analyst_prompt_editable(self) -> str:
        """Combined editable section for UI display."""
        return self.analyst_prompt_editable_part1 + "\n" + self.analyst_prompt_editable_part2
    
    @analyst_prompt_editable.setter
    def analyst_prompt_editable(self, value: str):
        """When setting combined editable section, split into part1 and part2."""
        value = value.strip()
        # Try to split at "INSTRUCTIONS" (look for it as a section header, typically on its own line)
        # Look for patterns like "\nINSTRUCTIONS\n" or start of string "INSTRUCTIONS\n"
        import re
        # Match "INSTRUCTIONS" as a standalone line (possibly with leading/trailing whitespace)
        pattern = r'(\n|^)(INSTRUCTIONS)\s*\n'
        match = re.search(pattern, value, re.MULTILINE)
        
        if match:
            split_pos = match.start() + len(match.group(1))  # Start after the newline (or at start)
            part1 = value[:split_pos].strip()
            part2 = value[split_pos:].strip()
            self.analyst_prompt_editable_part1 = part1
            self.analyst_prompt_editable_part2 = part2
        else:
            # If no INSTRUCTIONS marker found, try to preserve existing structure
            # Check if value contains both SYSTEM and DEFINITIONS but no INSTRUCTIONS
            if "SYSTEM" in value and "DEFINITIONS" in value and "INSTRUCTIONS" not in value:
                # User only edited part1, keep existing part2
                self.analyst_prompt_editable_part1 = value
                # Keep existing part2 unchanged
            else:
                # Fallback: put everything in part1, clear part2
                self.analyst_prompt_editable_part1 = value
                self.analyst_prompt_editable_part2 = ""

    # Internal storage for full prompt override (for backward compatibility)
    _analyst_prompt_override: str = ""
    
    # Full prompt (assembled from parts) - maintained for backward compatibility
    @property
    def analyst_prompt(self) -> str:
        """Assemble the full prompt from locked and editable sections."""
        # If there's an override (set directly), use it for backward compatibility
        if self._analyst_prompt_override:
            return self._analyst_prompt_override
        
        # Otherwise, assemble from parts in the correct order:
        # Header → Editable Part1 (SYSTEM + DEFINITIONS) → OUTPUT RULES → SCHEMA → EXAMPLES → Editable Part2 (INSTRUCTIONS) → Footer
        return (
            self._prompt_header +
            "\n" +
            self.analyst_prompt_editable_part1 +
            "\n" +
            self._prompt_output_rules +
            "\n" +
            self._prompt_schema +
            "\n" +
            self._prompt_examples +
            "\n" +
            self.analyst_prompt_editable_part2 +
            "\n" +
            self._prompt_footer
        )
    
    @analyst_prompt.setter
    def analyst_prompt(self, value: str):
        """When setting full prompt directly, store as override for backward compatibility."""
        # Store the full prompt as override
        # This allows old code that sets analyst_prompt directly to still work
        self._analyst_prompt_override = value


PROMPTS = AnalystPrompts()