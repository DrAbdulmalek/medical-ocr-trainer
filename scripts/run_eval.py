"""Run OCR evaluation suite — CLI entry point.

Usage:
    python scripts/run_eval.py evaluate --dataset data/golden/sample_eval_set.json
    python scripts/run_eval.py quick-eval --reference "hello world" --hypothesis "helo world"
    python scripts/run_eval.py create-dataset --terms-file terms.txt --output new_dataset.json
"""

import click
import json
import sys
from pathlib import Path


@click.group()
def cli():
    """Medical OCR Evaluation Toolkit.

    Evaluate OCR quality with CER, WER, medical term accuracy,
    and comprehensive benchmarking against golden datasets.
    """
    pass


@cli.command()
@click.option(
    "--dataset", required=True, type=click.Path(exists=True),
    help="Path to evaluation dataset JSON or CSV",
)
@click.option(
    "--output", default=None, type=click.Path(),
    help="Output report path (default: stdout)",
)
@click.option(
    "--format", "fmt", default="markdown", type=click.Choice(["markdown", "json"]),
    help="Output format (default: markdown)",
)
@click.option(
    "--engine", default="evaluated",
    help="Name of the OCR engine for report header (default: evaluated)",
)
def evaluate(dataset, output, fmt, engine):
    """Run evaluation against a golden dataset."""
    from evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner()
    result = runner.run(dataset, engine_name=engine)

    if fmt == "markdown":
        content = result.to_markdown()
    else:
        content = result.to_json()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Report written to {output}", err=True)
    else:
        click.echo(content)


@cli.command()
@click.option("--reference", required=True, help="Ground truth text")
@click.option("--hypothesis", required=True, help="OCR output text")
@click.option("--terms", multiple=True, help="Medical terms to check (repeatable)")
def quick_eval(reference, hypothesis, terms):
    """Quick evaluation of two text strings."""
    from evaluation.metrics import OCRMetrics

    metrics = OCRMetrics(medical_terms=list(terms) if terms else None)
    report = metrics.evaluate(reference, hypothesis)

    click.echo("=" * 50)
    click.echo("Quick OCR Evaluation")
    click.echo("=" * 50)
    click.echo(f"Reference:   {reference}")
    click.echo(f"Hypothesis: {hypothesis}")
    click.echo("-" * 50)
    click.echo(f"CER:                  {report['cer']:.4f}")
    click.echo(f"WER:                  {report['wer']:.4f}")
    click.echo(f"Overall Quality:      {report['overall_quality']:.2%}")
    click.echo("-" * 50)

    med = report["medical_term_accuracy"]
    if med["total_terms"] > 0:
        click.echo(f"Medical Terms:         {med['total_terms']} checked")
        click.echo(f"  Found:              {len(med['terms_found'])}")
        click.echo(f"  Partial:            {len(med['terms_partial'])}")
        click.echo(f"  Missing:            {len(med['terms_missing'])}")
        click.echo(f"  Accuracy:           {med['accuracy']:.2%}")

        for d in med["details"]:
            icon = {"found": "+", "partial": "~", "missing": "!"}.get(d["status"], "?")
            click.echo(f"    [{icon}] {d['term']} -> {d['status']}", err=True)
    else:
        click.echo("Medical Terms:         No terms provided (use --terms)")

    click.echo("=" * 50)


@cli.command()
@click.option(
    "--terms-file", required=True, type=click.Path(exists=True),
    help="File with medical terms (one per line)",
)
@click.option(
    "--output", required=True, type=click.Path(),
    help="Output JSON dataset path",
)
@click.option(
    "--name", default="new-dataset",
    help="Dataset name (default: new-dataset)",
)
def create_dataset(terms_file, output, name):
    """Create a new evaluation dataset template from a terms file."""
    from evaluation.dataset_manager import DatasetManager

    dm = DatasetManager()
    dm.create_template(name=name)

    with open(terms_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            term = line.strip()
            if term and not term.startswith("#"):
                dm.add_test_case({
                    "id": f"TC{line_num:03d}",
                    "source": "template",
                    "language": "en",
                    "reference": "",
                    "hypothesis": "",
                    "medical_terms": [term],
                    "category": "",
                })

    errors = dm.validate()
    if errors:
        click.echo(f"Warning: {len(errors)} validation issues:", err=True)
        for e in errors:
            click.echo(f"  - {e}", err=True)

    path = dm.save(output)
    stats = dm.statistics()
    click.echo(f"Created dataset: {path}")
    click.echo(f"  Test cases: {stats.get('total_cases', 0)}")
    click.echo(f"  Medical terms: {stats.get('total_medical_terms', 0)}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Fill in 'reference' and 'hypothesis' for each test case in {output}")
    click.echo(f"  2. Set 'language' and 'category' for each case")
    click.echo(f"  3. Run: python scripts/run_eval.py evaluate --dataset {output}")


@cli.command()
@click.option(
    "--dataset", required=True, type=click.Path(exists=True),
    help="Path to dataset JSON",
)
def stats(dataset):
    """Show statistics about an evaluation dataset."""
    from evaluation.dataset_manager import DatasetManager

    dm = DatasetManager()
    dm.load(dataset)
    s = dm.statistics()

    click.echo("=" * 50)
    click.echo(f"Dataset: {dm.data.get('name', 'unknown')}")
    click.echo(f"Version: {dm.data.get('version', 'unknown')}")
    click.echo("=" * 50)
    click.echo(f"Total test cases:    {s['total_cases']}")
    click.echo(f"Unique med terms:    {s['total_medical_terms']}")
    click.echo(f"Avg ref length:      {s['avg_reference_length']} chars")
    click.echo("")

    if s["languages"]:
        click.echo("Languages:")
        for lang, count in s["languages"].items():
            click.echo(f"  {lang}: {count}")

    if s["categories"]:
        click.echo("Categories:")
        for cat, count in s["categories"].items():
            click.echo(f"  {cat}: {count}")

    if s["sources"]:
        click.echo("Sources:")
        for src, count in s["sources"].items():
            click.echo(f"  {src}: {count}")

    click.echo("=" * 50)


@cli.command()
@click.option(
    "--dataset", required=True, type=click.Path(exists=True),
    help="Path to dataset JSON",
)
@click.option(
    "--output-dir", default=".",
    help="Directory for split outputs (default: current)",
)
@click.option(
    "--ratios", default="0.7,0.15,0.15",
    help="Split ratios: train,val,test (default: 0.7,0.15,0.15)",
)
@click.option(
    "--stratify", default=None,
    help="Field to stratify by (e.g., language, category)",
)
def split(dataset, output_dir, ratios, stratify):
    """Split dataset into train/val/test partitions."""
    from evaluation.dataset_manager import DatasetManager

    ratio_vals = [float(r) for r in ratios.split(",")]
    if len(ratio_vals) != 3:
        click.echo("Error: --ratios must have 3 values (e.g., 0.7,0.15,0.15)", err=True)
        sys.exit(1)

    dm = DatasetManager()
    dm.load(dataset)

    splits = dm.split(ratios=ratio_vals, stratify_by=stratify)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, partition in splits.items():
        path = out / f"{dm.data.get('name', 'dataset')}_{name}.json"
        dm_partition = DatasetManager()
        dm_partition.from_dict(partition)
        dm_partition.save(str(path))
        click.echo(f"  {name}: {len(partition['test_cases'])} cases -> {path}")

    click.echo("Split complete.")


if __name__ == "__main__":
    cli()
