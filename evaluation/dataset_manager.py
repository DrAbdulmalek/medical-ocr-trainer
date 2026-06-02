"""Golden Dataset Management for OCR Evaluation.

Manages evaluation datasets: loading, saving, validation,
splitting into train/val/test, and generating statistics.

Usage:
    from evaluation.dataset_manager import DatasetManager

    dm = DatasetManager()
    dm.load("data/golden/sample_eval_set.json")
    dm.validate()
    stats = dm.statistics()
    splits = dm.split(ratios=[0.6, 0.2, 0.2])
"""

import json
import random
from pathlib import Path
from typing import Optional


class DatasetManager:
    """Manage golden OCR evaluation datasets.

    Supports loading/saving datasets in JSON format, validating
    structure, splitting into train/val/test partitions, and
    computing descriptive statistics.
    """

    def __init__(self):
        self._data: dict = {}
        self._path: Optional[str] = None

    def load(self, path: str) -> dict:
        """Load a dataset from a JSON file.

        Args:
            path: Path to the JSON dataset file.

        Returns:
            The loaded dataset dictionary.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the JSON is invalid or structure is wrong.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        self._path = str(p)
        self.validate()
        return self._data

    def from_dict(self, data: dict) -> dict:
        """Load a dataset from a dictionary.

        Args:
            data: Dataset dictionary.

        Returns:
            The dataset dictionary after validation.
        """
        self._data = data
        self._path = None
        self.validate()
        return self._data

    @property
    def data(self) -> dict:
        """Return the current dataset dictionary."""
        return self._data

    def validate(self) -> list[str]:
        """Validate the current dataset structure.

        Checks for:
            - Required top-level keys (name, version, test_cases)
            - Required fields per test case (id, reference, hypothesis)
            - Unique test case IDs
            - Non-empty reference and hypothesis fields

        Returns:
            List of validation error messages (empty if valid).

        Raises:
            ValueError: If dataset has not been loaded.
        """
        if not self._data:
            raise ValueError("No dataset loaded. Call load() first.")

        errors = []

        # Check top-level keys
        for key in ("name", "version", "test_cases"):
            if key not in self._data:
                errors.append(f"Missing top-level key: '{key}'")

        if not isinstance(self._data.get("test_cases"), list):
            errors.append("'test_cases' must be a list")
            return errors

        # Check each test case
        ids_seen = set()
        for i, case in enumerate(self._data["test_cases"]):
            prefix = f"test_cases[{i}]"

            if not isinstance(case, dict):
                errors.append(f"{prefix}: must be a dictionary")
                continue

            for field in ("id", "reference", "hypothesis"):
                if field not in case:
                    errors.append(f"{prefix}: missing field '{field}'")

            # Check for duplicate IDs
            case_id = case.get("id", "")
            if case_id in ids_seen:
                errors.append(f"{prefix}: duplicate id '{case_id}'")
            ids_seen.add(case_id)

            # Check non-empty strings
            ref = case.get("reference", "")
            hyp = case.get("hypothesis", "")
            if not isinstance(ref, str) or not ref.strip():
                errors.append(f"{prefix}: 'reference' must be a non-empty string")
            if not isinstance(hyp, str) or not hyp.strip():
                errors.append(f"{prefix}: 'hypothesis' must be a non-empty string")

            # Validate medical_terms if present
            terms = case.get("medical_terms")
            if terms is not None and not isinstance(terms, list):
                errors.append(f"{prefix}: 'medical_terms' must be a list")

        return errors

    def save(self, path: str) -> str:
        """Save the current dataset to a JSON file.

        Args:
            path: Output file path.

        Returns:
            The path written to.

        Raises:
            ValueError: If no dataset is loaded.
        """
        if not self._data:
            raise ValueError("No dataset loaded. Call load() first.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

        self._path = str(p)
        return str(p)

    def statistics(self) -> dict:
        """Generate descriptive statistics about the dataset.

        Returns:
            Dictionary with statistics including:
                - total_cases: number of test cases
                - languages: language distribution
                - categories: category distribution
                - sources: source distribution
                - avg_reference_length: average reference text length
                - total_medical_terms: total unique medical terms
                - per_case_lengths: list of (id, ref_len, hyp_len) tuples
        """
        if not self._data:
            return {}

        cases = self._data.get("test_cases", [])
        if not cases:
            return {
                "total_cases": 0,
                "languages": {},
                "categories": {},
                "sources": {},
                "avg_reference_length": 0,
                "total_medical_terms": 0,
                "per_case_lengths": [],
            }

        languages = {}
        categories = {}
        sources = {}
        all_terms = set()
        per_case_lengths = []

        for case in cases:
            # Language distribution
            lang = case.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1

            # Category distribution
            cat = case.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

            # Source distribution
            src = case.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

            # Lengths
            ref_len = len(case.get("reference", ""))
            hyp_len = len(case.get("hypothesis", ""))
            per_case_lengths.append({
                "id": case.get("id", ""),
                "reference_length": ref_len,
                "hypothesis_length": hyp_len,
            })

            # Medical terms
            for term in case.get("medical_terms", []):
                all_terms.add(term.lower())

        total_ref_len = sum(
            len(case.get("reference", "")) for case in cases
        )

        return {
            "total_cases": len(cases),
            "languages": dict(sorted(languages.items())),
            "categories": dict(sorted(categories.items())),
            "sources": dict(sorted(sources.items())),
            "avg_reference_length": round(
                total_ref_len / len(cases), 1
            ),
            "total_medical_terms": len(all_terms),
            "per_case_lengths": per_case_lengths,
        }

    def split(
        self,
        ratios: list[float] = None,
        seed: int = 42,
        stratify_by: Optional[str] = None,
    ) -> dict[str, dict]:
        """Split the dataset into train/val/test partitions.

        Args:
            ratios: Split ratios [train, val, test].
                Defaults to [0.7, 0.15, 0.15].
            seed: Random seed for reproducibility.
            stratify_by: Field name to stratify by (e.g., 'language',
                'category'). If None, random split is used.

        Returns:
            Dictionary with keys 'train', 'val', 'test', each
            containing a dataset dictionary with the respective
            subset of test cases.
        """
        if ratios is None:
            ratios = [0.7, 0.15, 0.15]

        if len(ratios) != 3:
            raise ValueError("ratios must have exactly 3 values: [train, val, test]")

        if abs(sum(ratios) - 1.0) > 0.01:
            raise ValueError(f"ratios must sum to ~1.0, got {sum(ratios)}")

        cases = list(self._data.get("test_cases", []))
        if not cases:
            return {"train": {"test_cases": []}, "val": {"test_cases": []}, "test": {"test_cases": []}}

        rng = random.Random(seed)

        if stratify_by:
            # Group by stratification field
            groups: dict[str, list] = {}
            for case in cases:
                key = case.get(stratify_by, "unknown")
                groups.setdefault(key, []).append(case)

            # Split each group proportionally
            train, val, test = [], [], []
            for key, group in groups.items():
                rng.shuffle(group)
                n = len(group)
                n_train = max(1, int(n * ratios[0]))
                n_val = max(0, int(n * ratios[1]))
                train.extend(group[:n_train])
                val.extend(group[n_train:n_train + n_val])
                test.extend(group[n_train + n_val:])
        else:
            rng.shuffle(cases)
            n = len(cases)
            n_train = max(1, int(n * ratios[0]))
            n_val = max(0, int(n * ratios[1]))
            train = cases[:n_train]
            val = cases[n_train:n_train + n_val]
            test = cases[n_train + n_val:]

        def make_partition(cases_list: list, name: str) -> dict:
            return {
                "name": f"{self._data.get('name', 'dataset')}-{name}",
                "description": self._data.get("description", ""),
                "version": self._data.get("version", "1.0.0"),
                "split": name,
                "test_cases": cases_list,
            }

        return {
            "train": make_partition(train, "train"),
            "val": make_partition(val, "val"),
            "test": make_partition(test, "test"),
        }

    def add_test_case(self, case: dict) -> None:
        """Add a single test case to the dataset.

        Args:
            case: Test case dictionary with required fields:
                id, reference, hypothesis. Optional: language,
                source, medical_terms, category.
        """
        if not self._data:
            self._data = {
                "name": "untitled",
                "description": "",
                "version": "1.0.0",
                "test_cases": [],
            }

        self._data.setdefault("test_cases", []).append(case)

    def remove_test_case(self, case_id: str) -> bool:
        """Remove a test case by ID.

        Args:
            case_id: ID of the test case to remove.

        Returns:
            True if removed, False if not found.
        """
        cases = self._data.get("test_cases", [])
        original_len = len(cases)
        self._data["test_cases"] = [
            c for c in cases if c.get("id") != case_id
        ]
        return len(self._data["test_cases"]) < original_len

    def create_template(self, name: str = "new-dataset") -> dict:
        """Create an empty dataset template.

        Args:
            name: Dataset name.

        Returns:
            New dataset template dictionary.
        """
        self._data = {
            "name": name,
            "description": "",
            "version": "1.0.0",
            "test_cases": [],
        }
        return self._data
