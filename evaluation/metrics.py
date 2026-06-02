"""OCR Quality Evaluation Metrics.

Provides CER, WER, medical term accuracy, and custom scoring
for medical OCR output evaluation.

Usage:
    from evaluation.metrics import OCRMetrics

    metrics = OCRMetrics(medical_terms=["diabetes", "hypertension"])
    report = metrics.evaluate(
        reference="Patient has diabetes and hypertension",
        hypothesis="Patient has diabetse and hypertention"
    )
"""

import re
from typing import Optional


class OCRMetrics:
    """Calculate OCR quality metrics for medical documents.

    Attributes:
        _medical_terms: Normalized set of medical terms used for
            medical term accuracy evaluation.
    """

    def __init__(self, medical_terms: Optional[list[str]] = None):
        """Initialize OCRMetrics.

        Args:
            medical_terms: Optional list of medical terms to check
                preservation in OCR output. Terms are normalized to
                lowercase for case-insensitive matching.
        """
        self._medical_terms = set(t.lower() for t in (medical_terms or []))

    # ------------------------------------------------------------------
    # Core edit-distance algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings.

        Uses dynamic programming with O(m*n) time and space.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Integer edit distance.
        """
        m, n = len(s1), len(s2)
        # Optimize: use two rows instead of full matrix
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr[j] = min(
                    curr[j - 1] + 1,       # insertion
                    prev[j] + 1,            # deletion
                    prev[j - 1] + cost,     # substitution
                )
            prev = curr
        return prev[n]

    # ------------------------------------------------------------------
    # Public metric methods
    # ------------------------------------------------------------------

    def character_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate (CER).

        CER = edit_distance(reference, hypothesis) / len(reference)

        Characters are compared after stripping whitespace from both
        strings to measure transcription accuracy at the character level.

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text to evaluate.

        Returns:
            Float between 0.0 (perfect) and 1.0+ (poor).
            Returns 1.0 if reference is empty.
        """
        ref_chars = reference.replace(" ", "")
        hyp_chars = hypothesis.replace(" ", "")
        if not ref_chars:
            return 1.0
        return self._edit_distance(ref_chars, hyp_chars) / len(ref_chars)

    def word_error_rate(self, reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate (WER).

        WER = word_edit_distance(reference, hypothesis) / num_reference_words

        Words are split by whitespace.

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text to evaluate.

        Returns:
            Float between 0.0 (perfect) and 1.0+ (poor).
            Returns 1.0 if reference has no words (unless hypothesis is also empty).
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        return self._edit_distance(
            " ".join(ref_words), " ".join(hyp_words)
        ) / len(ref_words)

    def medical_term_accuracy(self, reference: str, hypothesis: str) -> dict:
        """Check how well medical terms are preserved in OCR output.

        For each known medical term, checks if it appears (with tolerance)
        in the hypothesis. Uses fuzzy matching with edit distance relative
        to term length.

        Args:
            reference: Ground truth text (used to identify expected terms).
            hypothesis: OCR output text to evaluate.

        Returns:
            Dictionary with:
                - terms_found: list of terms found in hypothesis
                - terms_missing: list of terms not found
                - terms_partial: list of partially matched terms
                - accuracy: fraction of reference terms found (0.0 - 1.0)
                - total_terms: number of known terms checked
                - details: per-term breakdown with edit distances
        """
        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()

        # Determine which medical terms are actually in the reference
        terms_in_ref = []
        for term in sorted(self._medical_terms):
            if term in ref_lower:
                terms_in_ref.append(term)

        terms_found = []
        terms_missing = []
        terms_partial = []
        details = []

        for term in terms_in_ref:
            if term in hyp_lower:
                # Exact match
                terms_found.append(term)
                details.append({
                    "term": term,
                    "status": "found",
                    "edit_distance": 0,
                })
            else:
                # Try fuzzy matching — check if a close variant exists
                # in the hypothesis
                best_dist = float("inf")
                best_match = ""
                for hyp_word in re.split(r"[\s,;.]+", hyp_lower):
                    if not hyp_word:
                        continue
                    # Check if the term is contained in a hypothesis word
                    if term in hyp_word or hyp_word in term:
                        dist = abs(len(term) - len(hyp_word))
                    else:
                        dist = self._edit_distance(term, hyp_word)

                    if dist < best_dist:
                        best_dist = dist
                        best_match = hyp_word

                tolerance = max(1, len(term) * 0.2)  # 20% tolerance

                if best_dist <= tolerance and best_match:
                    terms_partial.append(term)
                    details.append({
                        "term": term,
                        "status": "partial",
                        "edit_distance": best_dist,
                        "closest_match": best_match,
                    })
                else:
                    terms_missing.append(term)
                    details.append({
                        "term": term,
                        "status": "missing",
                        "edit_distance": best_dist if best_match else len(term),
                        "closest_match": best_match or None,
                    })

        total = len(terms_in_ref)
        accuracy = len(terms_found) / total if total > 0 else 1.0

        return {
            "terms_found": terms_found,
            "terms_missing": terms_missing,
            "terms_partial": terms_partial,
            "accuracy": accuracy,
            "total_terms": total,
            "details": details,
        }

    def evaluate(self, reference: str, hypothesis: str) -> dict:
        """Run all metrics and return comprehensive evaluation report.

        Args:
            reference: Ground truth text.
            hypothesis: OCR output text to evaluate.

        Returns:
            Dictionary with CER, WER, medical term accuracy, and
            an overall quality score.
        """
        cer = self.character_error_rate(reference, hypothesis)
        wer = self.word_error_rate(reference, hypothesis)
        med = self.medical_term_accuracy(reference, hypothesis)

        # Composite quality score: weighted average
        # CER and WER are inverted (1 - error_rate) and combined with
        # medical term accuracy
        cer_quality = max(0.0, 1.0 - cer)
        wer_quality = max(0.0, 1.0 - wer)
        med_quality = med["accuracy"]

        # Weighted composite: CER 35%, WER 35%, Medical 30%
        overall = (
            cer_quality * 0.35
            + wer_quality * 0.35
            + med_quality * 0.30
        )

        return {
            "cer": round(cer, 6),
            "wer": round(wer, 6),
            "medical_term_accuracy": med,
            "overall_quality": round(overall, 6),
            "reference_length": len(reference),
            "hypothesis_length": len(hypothesis),
        }
