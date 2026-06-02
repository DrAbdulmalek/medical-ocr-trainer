"""OCR Benchmark Runner.

Compare OCR engines against golden reference texts and produce reports.

Usage:
    from evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner()
    report = runner.run("data/golden/sample_eval_set.json")
    print(report.to_markdown())

CLI usage:
    python -m evaluation.benchmark --dataset data/golden/sample_eval_set.json
"""

import json
import csv
import io
import sys
from pathlib import Path
from typing import Optional

from evaluation.metrics import OCRMetrics


class BenchmarkResult:
    """Container for benchmark results and report generation."""

    def __init__(self, dataset_name: str, engine_name: str, results: list[dict]):
        """Initialize benchmark result.

        Args:
            dataset_name: Name of the evaluation dataset.
            engine_name: Name of the OCR engine being evaluated.
            results: List of per-test-case result dictionaries.
        """
        self.dataset_name = dataset_name
        self.engine_name = engine_name
        self.results = results

    def _aggregate(self) -> dict:
        """Compute aggregate statistics across all test cases."""
        if not self.results:
            return {
                "avg_cer": 0.0, "avg_wer": 0.0,
                "avg_medical_accuracy": 0.0, "avg_overall_quality": 0.0,
                "total_cases": 0,
            }

        total = len(self.results)
        avg_cer = sum(r["cer"] for r in self.results) / total
        avg_wer = sum(r["wer"] for r in self.results) / total
        avg_med = sum(
            r["medical_term_accuracy"]["accuracy"] for r in self.results
        ) / total
        avg_overall = sum(r["overall_quality"] for r in self.results) / total

        return {
            "avg_cer": round(avg_cer, 6),
            "avg_wer": round(avg_wer, 6),
            "avg_medical_accuracy": round(avg_med, 6),
            "avg_overall_quality": round(avg_overall, 6),
            "total_cases": total,
        }

    def to_markdown(self) -> str:
        """Generate a markdown report with tables.

        Returns:
            Complete markdown string with per-case table and summary.
        """
        lines = []
        lines.append(f"# OCR Benchmark Report")
        lines.append(f"")
        lines.append(f"**Dataset:** {self.dataset_name}")
        lines.append(f"**Engine:** {self.engine_name}")
        lines.append(f"**Test Cases:** {len(self.results)}")
        lines.append(f"")

        # Aggregate summary
        agg = self._aggregate()
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Avg CER | {agg['avg_cer']:.4f} |")
        lines.append(f"| Avg WER | {agg['avg_wer']:.4f} |")
        lines.append(f"| Avg Medical Accuracy | {agg['avg_medical_accuracy']:.2%} |")
        lines.append(f"| Avg Overall Quality | {agg['avg_overall_quality']:.2%} |")
        lines.append("")

        # Per-case table
        lines.append("## Per-Case Results")
        lines.append("")
        lines.append("| ID | Category | Language | CER | WER | Medical Acc | Overall |")
        lines.append("|----|----------|----------|-----|-----|-------------|---------|")

        for r in self.results:
            med_acc = r["medical_term_accuracy"]["accuracy"]
            lines.append(
                f"| {r['id']} | {r.get('category', '-')} | "
                f"{r.get('language', '-')} | "
                f"{r['cer']:.4f} | {r['wer']:.4f} | "
                f"{med_acc:.2%} | {r['overall_quality']:.2%} |"
            )

        lines.append("")

        # Failed medical terms detail
        missing_terms = []
        for r in self.results:
            med = r["medical_term_accuracy"]
            for d in med.get("details", []):
                if d["status"] in ("missing", "partial"):
                    missing_terms.append((r["id"], d))

        if missing_terms:
            lines.append("## Medical Term Issues")
            lines.append("")
            lines.append("| Test Case | Term | Status | Edit Dist | Closest |")
            lines.append("|-----------|------|--------|-----------|---------|")
            for case_id, d in missing_terms:
                closest = d.get("closest_match", "-")
                lines.append(
                    f"| {case_id} | {d['term']} | {d['status']} | "
                    f"{d['edit_distance']} | {closest} |"
                )
            lines.append("")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serialize results to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON formatted string.
        """
        return json.dumps({
            "dataset": self.dataset_name,
            "engine": self.engine_name,
            "aggregate": self._aggregate(),
            "cases": self.results,
        }, indent=indent, ensure_ascii=False)


class BenchmarkRunner:
    """Run OCR evaluation benchmarks against golden datasets.

    Loads test cases from JSON files, computes metrics for each case,
    and generates comprehensive reports.

    Usage:
        runner = BenchmarkRunner(medical_terms=["diabetes", "hypertension"])
        result = runner.run("data/golden/sample_eval_set.json")
        print(result.to_markdown())
    """

    def __init__(self, medical_terms: Optional[list[str]] = None):
        """Initialize BenchmarkRunner.

        Args:
            medical_terms: Optional list of medical terms to check
                in medical term accuracy. If not provided, terms
                are extracted from the dataset itself.
        """
        self._medical_terms = [t.lower() for t in (medical_terms or [])]
        self._metrics = OCRMetrics(medical_terms=self._medical_terms)

    @staticmethod
    def load_dataset(path: str) -> dict:
        """Load an evaluation dataset from JSON file.

        Args:
            path: Path to JSON dataset file.

        Returns:
            Dataset dictionary with test_cases list.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If dataset structure is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        if "test_cases" not in data:
            raise ValueError(
                "Invalid dataset: missing 'test_cases' key"
            )

        for i, case in enumerate(data["test_cases"]):
            for field in ("id", "reference", "hypothesis"):
                if field not in case:
                    raise ValueError(
                        f"Test case {i} missing required field: {field}"
                    )

        return data

    @staticmethod
    def load_dataset_csv(path: str) -> dict:
        """Load an evaluation dataset from CSV file.

        CSV columns: id, reference, hypothesis, medical_terms, category, language

        Args:
            path: Path to CSV dataset file.

        Returns:
            Dataset dictionary compatible with JSON format.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        test_cases = []
        with open(p, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                medical_terms = (
                    [t.strip() for t in row.get("medical_terms", "").split(",")
                     if t.strip()]
                    if row.get("medical_terms") else []
                )
                test_cases.append({
                    "id": row.get("id", ""),
                    "source": row.get("source", "csv"),
                    "language": row.get("language", "en"),
                    "reference": row.get("reference", ""),
                    "hypothesis": row.get("hypothesis", ""),
                    "medical_terms": medical_terms,
                    "category": row.get("category", ""),
                })

        return {
            "name": p.stem,
            "description": f"Loaded from {p.name}",
            "version": "1.0.0",
            "test_cases": test_cases,
        }

    def run(
        self,
        dataset_path: str,
        engine_name: str = "evaluated",
    ) -> BenchmarkResult:
        """Run evaluation against a golden dataset.

        Args:
            dataset_path: Path to JSON or CSV dataset file.
            engine_name: Name of the OCR engine (for reporting).

        Returns:
            BenchmarkResult with per-case metrics and aggregates.
        """
        # Auto-detect format
        if dataset_path.endswith(".csv"):
            dataset = self.load_dataset_csv(dataset_path)
        else:
            dataset = self.load_dataset(dataset_path)

        # Collect all medical terms from dataset if none provided
        all_terms = list(self._medical_terms)
        for case in dataset["test_cases"]:
            for term in case.get("medical_terms", []):
                if term.lower() not in [t.lower() for t in all_terms]:
                    all_terms.append(term)

        # Re-create metrics with full term list
        metrics = OCRMetrics(medical_terms=all_terms)

        results = []
        for case in dataset["test_cases"]:
            report = metrics.evaluate(
                reference=case["reference"],
                hypothesis=case["hypothesis"],
            )
            report["id"] = case["id"]
            report["category"] = case.get("category", "")
            report["language"] = case.get("language", "")
            report["source"] = case.get("source", "")
            results.append(report)

        return BenchmarkResult(
            dataset_name=dataset.get("name", "unknown"),
            engine_name=engine_name,
            results=results,
        )


def main():
    """CLI entry point for running benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OCR evaluation benchmarks"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to evaluation dataset (JSON or CSV)"
    )
    parser.add_argument(
        "--engine", default="evaluated",
        help="Name of the OCR engine (default: evaluated)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown",
        help="Output format (default: markdown)"
    )
    args = parser.parse_args()

    runner = BenchmarkRunner()
    result = runner.run(args.dataset, engine_name=args.engine)

    if args.format == "markdown":
        content = result.to_markdown()
    else:
        content = result.to_json()

    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(content)


if __name__ == "__main__":
    main()
