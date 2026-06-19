# Multi-Modal Evidence Review System

This folder contains the runnable solution for the HackerRank Orchestrate
multi-modal evidence review challenge. The system verifies damage claims for
cars, laptops, and packages using the claim conversation, submitted images,
user history, and evidence requirements.

## Quick Run

From the repository root:

```bash
cd code
pip install -r requirements.txt
python main.py --output ../output.csv
```

To run the labeled sample set instead:

```bash
cd code
python main.py --sample --output ../sample_output.csv
```

To generate the evaluation report:

```bash
cd code
python evaluation/main.py
```

The final HackerRank prediction file is written at:

```text
../output.csv
```

## API Setup

Secrets are read only from environment variables or from a local `.env` file at
the repository root. Do not commit `.env`.

Create a local env file:

```bash
cp .env.example .env
```

Then paste a Google AI Studio key:

```env
GOOGLE_API_KEY=your_key_here
GEMINI_TEXT_MODEL=gemini-3.1-flash-lite
GEMINI_VISION_MODEL=gemini-3.1-flash-lite
```

OpenAI is supported only as an optional fallback:

```env
OPENAI_API_KEY=
OPENAI_TEXT_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o-mini
```

The submitted configuration uses Gemini API calls only; there is no local
inference runtime, Ollama dependency, or heavyweight model dependency.

## How The Pipeline Works

The solution is a staged orchestration pipeline:

```text
Claim Input
  -> Claim Extraction
  -> Vision Analysis
  -> Evidence Requirement Validation
  -> Conflict Detection
  -> Risk Assessment
  -> Decision Engine
  -> Output Validation
  -> output.csv
```

### 1. Claim Extraction

`pipeline/claim_extraction.py` reads the customer conversation and extracts the
actual claimed damage:

- issue type, such as `dent`, `scratch`, `crack`, or `water_damage`
- object part, such as `front_bumper`, `screen`, or `box`
- a short summary of what the customer wants reviewed

This is text-only and does not look at the images.

### 2. Vision Analysis

`pipeline/vision_analysis.py` sends the submitted images for each claim to the
VLM in one batched call per claim. The prompt tells the model to inspect only
what is visible in the images and return structured JSON for each image:

- whether the image is valid/corrupt
- object type and object-part match
- visible issue type
- severity
- manipulation or text-in-image flags
- image-specific notes

The claim narrative is intentionally excluded from the vision prompt so the
model does not see the user's allegation while judging the image.

### 3. Evidence Requirement Validation

`pipeline/evidence_validator.py` checks the image analysis against
`dataset/evidence_requirements.csv`. This determines whether the submitted
image set is sufficient for automated review.

### 4. Conflict Detection

`pipeline/conflict_detection.py` compares the extracted claim with visible
image evidence. It flags cases where the user claims one damage/part but the
images show a different part, no visible damage, wrong object, or other
contradiction.

### 5. Risk Assessment

`pipeline/risk_assessment.py` uses image quality flags, manipulation signals,
and `dataset/user_history.csv` to add risk flags such as:

- `blurry_image`
- `wrong_object`
- `damage_not_visible`
- `claim_mismatch`
- `user_history_risk`
- `manual_review_required`

### 6. Decision Engine

`pipeline/decision_engine.py` combines the extracted claim, vision analysis,
evidence checks, conflicts, and risk signals to produce the final claim status:

- `supported`
- `contradicted`
- `not_enough_information`

The decision logic is guided by `playbook/claim_review_playbook.yaml`.

### 7. Output Validation

`pipeline/output_validator.py` normalizes the output to the exact allowed
values and required CSV schema before `main.py` writes the final file.

## Output Schema

`schemas.py` defines the exact output columns. The final `output.csv` must have
one row per row in `dataset/claims.csv` and these columns in this order:

```text
user_id
image_paths
user_claim
claim_object
evidence_standard_met
evidence_standard_met_reason
risk_flags
issue_type
object_part
claim_status
claim_status_justification
supporting_image_ids
valid_image
severity
```

## Evaluation

The evaluation workflow runs against `dataset/sample_claims.csv`, compares
predictions to the labeled sample fields, and writes:

```text
../evaluation/evaluation_report.md
```

The report includes:

- claim-status accuracy
- per-field accuracy
- approach comparison
- model calls and image usage
- token and cost estimates
- runtime and quota considerations
- final architecture justification

The latest measured sample result is:

```text
Claim-status accuracy: 75.0% (15/20)
```

## Important Files

```text
code/main.py                         Terminal entry point
code/config.py                       Environment and path configuration
code/models/client.py                Gemini/OpenAI API clients, cache, retry logic
code/pipeline/claim_extraction.py    Text claim extraction
code/pipeline/vision_analysis.py     Batched VLM image analysis
code/pipeline/evidence_validator.py  Evidence requirement checks
code/pipeline/conflict_detection.py  Claim/image mismatch detection
code/pipeline/risk_assessment.py     Risk flag calculation
code/pipeline/decision_engine.py     Final claim-status decision
code/pipeline/output_validator.py    Schema/value normalization
code/playbook/claim_review_playbook.yaml
code/evaluation/main.py              Evaluation workflow
```

## Reproducibility Notes

- Model responses are cached in `code/.cache/` by prompt, model, and image
  path. This speeds up repeated local checks.
- `code/.cache/` is not required in the submission zip and is intentionally
  excluded.
- The system has deterministic validation and decision layers around the model
  outputs.
- The VLM is used for visual evidence extraction; final decisions are made by
  the pipeline, not by a single free-form model answer.

## Submission Checklist

Before submitting:

```bash
cd code
python main.py --output ../output.csv
python evaluation/main.py
```

Confirm:

- `../output.csv` has 44 rows for the provided `dataset/claims.csv`
- `../output.csv` columns match `schemas.OUTPUT_COLUMNS`
- `../evaluation/evaluation_report.md` exists
- `code.zip` includes this `code/` folder and the `evaluation/` folder
- `.env`, `.venv`, caches, and bytecode are excluded from the zip
