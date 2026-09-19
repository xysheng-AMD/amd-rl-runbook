"""Concept Snapshot Extractor — §4.5: auto-extract conceptual knowledge from reasoning."""



import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConceptSnapshot:
    concept_text: str
    inferred_question: str
    concept_count: int
    has_unverified_constants: bool = False


# Patterns indicating conceptual framework content
_MULTI_DEF_PAT = re.compile(
    r"(?:[•\-\*]\s*.+?(?:是|为|指|means|refers to|controls|maps to).+?\n){2,}",
    re.MULTILINE,
)
_CLASSIFICATION_PAT = re.compile(
    r"(?:分为|分成|分类|有以下|包括以下|three types|two categories|主要有)"
    r".*?(?:[:：]\s*\n|——)",
    re.IGNORECASE,
)
_PRINCIPLE_PAT = re.compile(
    r"(?:这是因为|原因是|所以|therefore|because|this is due to|the reason)"
    r".{20,}",
    re.IGNORECASE,
)

# Precise constant patterns that need source annotation
_PRECISE_CONST_PAT = re.compile(
    r"(?:0x[0-9a-fA-F]+|\b(?:bit|offset|position)\s*=?\s*\d+|\b\d{2,}(?:B|KB|MB|GB)\b)",
)
_VERIFY_ANNOTATION_PAT = re.compile(
    r"verify against|验证|check against|参考|see .+ for exact|确认", re.IGNORECASE
)


class ConceptExtractor:
    """Scans reasoning blocks and extracts concept snapshots."""

    def __init__(self, min_concepts: int = 2) -> None:
        self.min_concepts = min_concepts

    def extract(
        self, reasoning_text: str, segment_context: str = ""
    ) -> Optional[ConceptSnapshot]:
        """Extract concept snapshot from a reasoning block.

        Returns None if the block doesn't contain enough conceptual content.
        """
        if len(reasoning_text) < 100:
            return None

        concept_count = 0

        # Check for multi-definition pattern — count individual definition lines
        def_lines = re.findall(
            r"^[•\-\*]\s+.+?(?:是|为|指|means|refers to|controls|maps to).+",
            reasoning_text, re.MULTILINE,
        )
        concept_count += len(def_lines)

        # Check for classification structure
        if _CLASSIFICATION_PAT.search(reasoning_text):
            concept_count += 1

        # Check for principle explanation
        if _PRINCIPLE_PAT.search(reasoning_text):
            concept_count += 1

        # Count bullet points as potential concept definitions
        bullets = re.findall(r"^[\-\*•]\s+\*\*(.+?)\*\*", reasoning_text, re.MULTILINE)
        concept_count += len(bullets)

        if concept_count < self.min_concepts:
            return None

        # Check for precise constants without source annotation
        has_unverified = False
        const_matches = _PRECISE_CONST_PAT.finditer(reasoning_text)
        for match in const_matches:
            # Check if there's a verify annotation nearby (within 200 chars)
            start = max(0, match.start() - 200)
            end = min(len(reasoning_text), match.end() + 200)
            context_window = reasoning_text[start:end]
            if not _VERIFY_ANNOTATION_PAT.search(context_window):
                has_unverified = True
                break

        question = self._infer_question(reasoning_text, segment_context)

        return ConceptSnapshot(
            concept_text=reasoning_text,
            inferred_question=question,
            concept_count=concept_count,
            has_unverified_constants=has_unverified,
        )

    def _infer_question(self, reasoning: str, context: str) -> str:
        """Infer a plausible user question from the reasoning content."""
        # Try to find the topic from the first sentence
        first_line = reasoning.split("\n")[0].strip()

        # Look for key topics
        topics = {
            "cache": "How does cache policy work on gfx950?",
            "MFMA": "What are the MFMA instruction characteristics?",
            "LDS": "How does the LDS subsystem work?",
            "occupancy": "How is occupancy determined?",
            "roofline": "How to analyze kernel performance using the roofline model?",
            "tiling": "How does tile-based kernel optimization work?",
            "memory": "How does the memory hierarchy work?",
            "dtype": "What data type conversions are supported?",
            "pipeline": "How does software pipelining work?",
        }

        for keyword, question in topics.items():
            if keyword.lower() in reasoning.lower():
                return question

        # Fallback: generic question from first line
        if len(first_line) > 10:
            topic = first_line[:100].rstrip(".,;:")
            return f"Explain: {topic}"

        return "Explain the concept described in the analysis."
