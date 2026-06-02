

---

## Repository Status

| Field | Value |
|-------|-------|
| **Role** | Data Collection & Training Tool |
| **Status** | Active Development |
| **Layer** | Applications (Product) |
| **Priority** | Medium |
| **Relation** | Feeds training data to medical-handwriting-ocr and omni-medical-suite |

## Who Should Use This

- ML engineers building **medical handwriting training datasets**
- Clinical teams needing **human-in-the-loop correction** workflows
- Researchers comparing **multiple OCR engines** on medical documents
- Projects needing **active learning** data pipelines

## What This Produces

```
┌─────────────────────────┐
│   medical-ocr-trainer    │
│                         │
│   Input: Scanned images │
│   Output:               │
│   ├── JSONL (HF format) │──▶ medical-handwriting-ocr (fine-tuning)
│   ├── CSV datasets      │──▶ omni-medical-suite (evaluation)
│   ├── HF Image Folders  │──▶ HuggingFace Hub
│   └── Gold/Pending/      │
│       Rejected labels   │
└─────────────────────────┘
```

## When to Use This vs Other Repos

| Need | Repository |
|------|-----------|
| Collect & correct training data | **This repo** (medical-ocr-trainer) |
| Production OCR deployment | [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) |
| OCR correction engine | [medical-ocr-postprocessor](https://github.com/DrAbdulmalek/medical-ocr-postprocessor) |
| Unified platform | [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) |
| HF demo deployment | [medical-ocr-trainer-hf](https://github.com/DrAbdulmalek/medical-ocr-trainer-hf) |

## Related Repositories

| Repo | Role | Status |
|------|------|--------|
| [medical-handwriting-ocr](https://github.com/DrAbdulmalek/medical-handwriting-ocr) | Production OCR | Active |
| [medical-ocr-trainer-hf](https://github.com/DrAbdulmalek/medical-ocr-trainer-hf) | HF Deployment | Deployment |
| [omni-medical-suite](https://github.com/DrAbdulmalek/omni-medical-suite) | Main Platform | Active |
| [medical-ocr-postprocessor](https://github.com/DrAbdulmalek/medical-ocr-postprocessor) | Core Correction Engine | Active |

**License: MIT** — Dr. Abdulmalek Tamer Al-husseini
