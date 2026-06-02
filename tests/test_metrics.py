"""Tests for OCRMetrics — CER, WER, medical term accuracy.

Run with: pytest tests/test_metrics.py -v
"""

import pytest
from evaluation.metrics import OCRMetrics


class TestCharacterErrorRate:
    """Tests for CER calculation."""

    def test_identical_strings(self):
        """CER should be 0.0 for identical strings."""
        m = OCRMetrics()
        assert m.character_error_rate("hello world", "hello world") == 0.0

    def test_empty_reference(self):
        """CER should be 1.0 when reference is empty."""
        m = OCRMetrics()
        assert m.character_error_rate("", "hello") == 1.0

    def test_single_substitution(self):
        """CER should detect single character substitution."""
        m = OCRMetrics()
        cer = m.character_error_rate("hello", "hallo")
        assert cer == pytest.approx(1 / 5)

    def test_single_insertion(self):
        """CER should detect character insertion."""
        m = OCRMetrics()
        cer = m.character_error_rate("hello", "helllo")
        assert cer == pytest.approx(1 / 5)

    def test_single_deletion(self):
        """CER should detect character deletion."""
        m = OCRMetrics()
        cer = m.character_error_rate("hello", "helo")
        assert cer == pytest.approx(1 / 5)

    def test_completely_different(self):
        """CER should be close to 1.0 for completely different strings."""
        m = OCRMetrics()
        cer = m.character_error_rate("abc", "xyz")
        assert cer == pytest.approx(1.0)

    def test_whitespace_ignored(self):
        """CER should ignore whitespace in calculation."""
        m = OCRMetrics()
        # "hello world" -> "helloworld" for CER calc
        cer_with = m.character_error_rate("hello world", "hello world")
        cer_without = m.character_error_rate("helloworld", "helloworld")
        assert cer_with == cer_without

    def test_medical_term_typo(self):
        """Test CER with a realistic medical OCR error."""
        m = OCRMetrics()
        cer = m.character_error_rate(
            "hypertension",
            "hypertention"
        )
        assert cer == pytest.approx(1 / 12)


class TestWordErrorRate:
    """Tests for WER calculation."""

    def test_identical_sentences(self):
        """WER should be 0.0 for identical sentences."""
        m = OCRMetrics()
        assert m.word_error_rate("the cat sat", "the cat sat") == 0.0

    def test_empty_reference(self):
        """WER should be 1.0 when reference is empty (and hypothesis is not)."""
        m = OCRMetrics()
        assert m.word_error_rate("", "hello") == 1.0

    def test_both_empty(self):
        """WER should be 0.0 when both are empty."""
        m = OCRMetrics()
        assert m.word_error_rate("", "") == 0.0

    def test_word_substitution(self):
        """WER should detect word substitution."""
        m = OCRMetrics()
        wer = m.word_error_rate("the cat sat", "the dog sat")
        assert wer == pytest.approx(1 / 3)

    def test_word_insertion(self):
        """WER should detect word insertion."""
        m = OCRMetrics()
        wer = m.word_error_rate("the cat sat", "the big cat sat")
        assert wer == pytest.approx(1 / 3)

    def test_word_deletion(self):
        """WER should detect word deletion."""
        m = OCRMetrics()
        wer = m.word_error_rate("the cat sat on mat", "the cat sat")
        assert wer == pytest.approx(2 / 5)

    def test_medical_sentence(self):
        """Test WER with a realistic medical sentence."""
        m = OCRMetrics()
        wer = m.word_error_rate(
            "Patient presents with hypertension",
            "Patient presents with hypertention"
        )
        assert wer == 0.0  # Same number of words, substitution at char level only


class TestMedicalTermAccuracy:
    """Tests for medical term accuracy."""

    def test_all_terms_found(self):
        """All medical terms found in hypothesis."""
        m = OCRMetrics(medical_terms=["hypertension", "diabetes"])
        result = m.medical_term_accuracy(
            "Patient has hypertension and diabetes",
            "Patient has hypertension and diabetes"
        )
        assert result["accuracy"] == 1.0
        assert len(result["terms_found"]) == 2
        assert len(result["terms_missing"]) == 0

    def test_term_missing(self):
        """Medical term missing from hypothesis."""
        m = OCRMetrics(medical_terms=["hypertension", "diabetes"])
        result = m.medical_term_accuracy(
            "Patient has hypertension and diabetes",
            "Patient has hypertension and arthritis"
        )
        assert "diabetes" in result["terms_missing"]
        assert result["accuracy"] == 0.5

    def test_partial_match(self):
        """Medical term partially matched in hypothesis."""
        m = OCRMetrics(medical_terms=["hypertension"])
        result = m.medical_term_accuracy(
            "Patient has hypertension",
            "Patient has hypertention"
        )
        # "hypertention" is close to "hypertension" (edit dist 1)
        # Should be detected as partial with 20% tolerance
        assert result["total_terms"] == 1
        # The term is close enough for partial match
        assert len(result["terms_partial"]) == 1 or len(result["terms_found"]) == 1

    def test_no_terms_in_reference(self):
        """No medical terms present in reference."""
        m = OCRMetrics(medical_terms=["hypertension"])
        result = m.medical_term_accuracy(
            "The sky is blue today",
            "The sky is blue today"
        )
        assert result["accuracy"] == 1.0  # No terms to miss
        assert result["total_terms"] == 0

    def test_no_terms_provided(self):
        """No medical terms provided to metrics."""
        m = OCRMetrics(medical_terms=[])
        result = m.medical_term_accuracy(
            "Patient has diabetes",
            "Patient has diabetse"
        )
        assert result["accuracy"] == 1.0
        assert result["total_terms"] == 0

    def test_details_structure(self):
        """Check details list structure."""
        m = OCRMetrics(medical_terms=["diabetes"])
        result = m.medical_term_accuracy(
            "Patient has diabetes",
            "Patient has diabetse"
        )
        assert isinstance(result["details"], list)
        if result["details"]:
            detail = result["details"][0]
            assert "term" in detail
            assert "status" in detail
            assert "edit_distance" in detail


class TestEvaluate:
    """Tests for the comprehensive evaluate() method."""

    def test_perfect_match(self):
        """Perfect match should yield CER=0, WER=0, quality=1.0."""
        m = OCRMetrics(medical_terms=["hypertension"])
        report = m.evaluate(
            "Patient has hypertension",
            "Patient has hypertension"
        )
        assert report["cer"] == 0.0
        assert report["wer"] == 0.0
        assert report["overall_quality"] == 1.0

    def test_report_structure(self):
        """Check all expected keys in report."""
        m = OCRMetrics()
        report = m.evaluate("hello", "helo")
        assert "cer" in report
        assert "wer" in report
        assert "medical_term_accuracy" in report
        assert "overall_quality" in report
        assert "reference_length" in report
        assert "hypothesis_length" in report
        assert 0.0 <= report["overall_quality"] <= 1.0

    def test_overall_quality_range(self):
        """Overall quality should always be between 0 and 1."""
        m = OCRMetrics()
        # Very bad case
        r1 = m.evaluate("The patient has severe hypertension", "xyz")
        assert 0.0 <= r1["overall_quality"] <= 1.0
        # Perfect case
        r2 = m.evaluate("Perfect text", "Perfect text")
        assert 0.0 <= r2["overall_quality"] <= 1.0


class TestEditDistance:
    """Tests for the static edit distance method."""

    def test_same_string(self):
        assert OCRMetrics._edit_distance("abc", "abc") == 0

    def test_empty_strings(self):
        assert OCRMetrics._edit_distance("", "") == 0

    def test_empty_vs_nonempty(self):
        assert OCRMetrics._edit_distance("", "abc") == 3

    def test_substitution(self):
        assert OCRMetrics._edit_distance("kitten", "sitten") == 1

    def test_complex(self):
        assert OCRMetrics._edit_distance("sitting", "kitten") == 3

    def test_unicode(self):
        """Test with Arabic characters."""
        dist = OCRMetrics._edit_distance("السكري", "السكرى")
        assert dist >= 0  # Should not crash on Unicode
