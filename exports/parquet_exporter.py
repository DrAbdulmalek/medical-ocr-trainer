"""
Parquet Exporter for Medical OCR Training Data
================================================
Exports training data to Parquet format for efficient storage and
compatibility with data processing frameworks (pandas, Polars, DuckDB).

Author: Dr. Abdulmalek
Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ParquetExporter:
    """
    Export medical OCR training data to Parquet format.
    
    Parquet provides:
    - Columnar storage with compression (10x smaller than JSONL)
    - Schema enforcement
    - Efficient partial reads (read only needed columns)
    - Full compatibility with pandas, Polars, DuckDB, Spark
    """

    def __init__(self, compression: str = "zstd"):
        """
        Args:
            compression: Compression codec. Options: snappy, gzip, zstd, lz4, brotli, none
        """
        self.compression = compression
        self._pa = None
        self._pd = None
        self._init_libraries()

    def _init_libraries(self):
        """Lazily initialize optional libraries."""
        try:
            import pyarrow as pa
            self._pa = pa
        except ImportError:
            logger.warning("pyarrow not installed. Install with: pip install pyarrow")
        
        try:
            import pandas as pd
            self._pd = pd
        except ImportError:
            logger.warning("pandas not installed. Install with: pip install pandas")

    def export(
        self,
        data: List[Dict[str, Any]],
        output_path: str,
        schema_overrides: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Export training data to Parquet.
        
        Args:
            data: List of dictionaries (training pairs or benchmark results)
            output_path: Output file path (.parquet)
            schema_overrides: Optional column type overrides
            
        Returns:
            Export statistics
        """
        if not data:
            logger.warning("No data to export")
            return {"rows": 0, "file": output_path}

        if self._pd is None or self._pa is None:
            logger.error("Both pandas and pyarrow are required for Parquet export")
            return {"rows": 0, "file": output_path, "error": "missing_dependencies"}

        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq

            df = pd.DataFrame(data)

            # Apply schema overrides if provided
            if schema_overrides:
                for col, dtype in schema_overrides.items():
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(dtype)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not cast column {col} to {dtype}: {e}")

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Write Parquet
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                output_path,
                compression=self.compression,
                engine="pyarrow",
            )

            file_size = Path(output_path).stat().st_size
            original_size = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            compression_ratio = original_size / max(file_size, 1)

            logger.info(
                f"Exported {len(data)} rows to {output_path} "
                f"({file_size / 1024:.1f} KB, {compression_ratio:.1f}x compression)"
            )

            return {
                "rows": len(data),
                "columns": list(df.columns),
                "file": output_path,
                "size_bytes": file_size,
                "compression_ratio": round(compression_ratio, 2),
                "compression_codec": self.compression,
            }

        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            return {"rows": 0, "file": output_path, "error": str(e)}

    def export_training_pairs(
        self,
        pairs: List[Dict[str, str]],
        output_path: str,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        Export OCR training pairs to Parquet.
        
        Each pair has:
        - input: OCR output (before correction)
        - target: Ground truth (correct text)
        - Optional: source, language, specialty, cer
        """
        return self.export(pairs, output_path)

    def export_benchmark_results(
        self,
        results: Dict[str, Any],
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Export benchmark results to Parquet for historical tracking.
        
        Converts nested results to flat rows for efficient querying.
        """
        rows = []
        for engine_name, engine_data in results.get("engines", {}).items():
            row = {
                "engine": engine_name,
                "cer": engine_data.get("cer", 0),
                "wer": engine_data.get("wer", 0),
                "medical_accuracy": engine_data.get("medical_accuracy", 0),
                "latency_ms": engine_data.get("latency_ms", 0),
                "throughput_pages_per_sec": engine_data.get("throughput", 0),
                "timestamp": results.get("timestamp", ""),
                "dataset": results.get("dataset", ""),
            }
            
            # Add per-language breakdowns
            for lang, lang_data in engine_data.get("by_language", {}).items():
                lang_row = {**row, "language": lang}
                lang_row["cer"] = lang_data.get("cer", row["cer"])
                lang_row["wer"] = lang_data.get("wer", row["wer"])
                lang_row["medical_accuracy"] = lang_data.get("medical_accuracy", row["medical_accuracy"])
                rows.append(lang_row)
            
            if not engine_data.get("by_language"):
                rows.append(row)

        return self.export(rows, output_path)

    def read(self, path: str, columns: Optional[List[str]] = None) -> "pd.DataFrame":
        """Read Parquet file, optionally selecting specific columns."""
        if self._pd is None:
            raise ImportError("pandas is required to read Parquet files")
        import pandas as pd
        return pd.read_parquet(path, columns=columns)