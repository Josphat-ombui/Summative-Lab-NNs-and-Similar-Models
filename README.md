# EcoSort: Intelligent Waste Management Assistant

Summative lab for the Moringa School Data Science programme (DSF-PT-15), Module 8 -
"Neural Networks and Similar Models." The notebook builds an integrated assistant for
Metro City's waste management department that identifies waste material from resident
photos, classifies waste from text descriptions, and generates recycling instructions
grounded in municipal policy, with a safeguard that never shows a resident unverified text.

## Contents

| File | Description |
|---|---|
| `waste_management_summative_BEST.ipynb` | The master notebook. All five rubric parts, run top to bottom. |
| `waste_descriptions.csv` | 5,000+ generated resident waste descriptions with category labels. |
| `waste_policy_documents.json` | Metro City policy documents used by the RAG component (Part 4). |
| RealWaste image folders | Not included in this repository; see Data below. |

## Project structure

The notebook is organised into five parts, matching the assignment rubric exactly (see
the rubric-alignment table in the notebook's own introduction cell for the full mapping):

0. **Environment setup and run configuration** - fixed seeds, device detection
   (CPU/GPU), and the epoch/batch-size settings used throughout, collected in one place
   so they're easy to adjust for a slower machine.
1. **Dataset exploration and preparation** - a full-dataset image audit (4,752 images:
   integrity, resolution, lighting, sharpness, background, and a perceptual-hash
   duplicate check), text and policy-document exploration, and a stratified 70/15/15
   train/validation/test split with no overlap between splits.
2. **Waste material classification (CNN)** - a 4-configuration architecture sweep
   (EfficientNetB0 and MobileNetV2 heads of varying width/depth/dropout), the winning
   configuration trained to convergence and then fine-tuned, with class weighting for
   the imbalanced categories and a full confusion-matrix and confidence-threshold
   analysis. Test accuracy: 0.8808 (macro-F1 0.8865).
3. **Waste description classification (text)** - a TF-IDF + Logistic Regression
   baseline and a fine-tuned DistilBERT model, both evaluated honestly: a near-duplicate
   leakage check (19.7% of test items have a close training neighbour) and a 16-item
   hand-written challenge set are used to explain the near-perfect headline accuracy
   rather than take it at face value.
4. **Recycling instruction generation (RAG)** - section-aware policy chunking, a
   3-way retrieval benchmark (two dense embedding variants plus a TF-IDF baseline), a
   fine-tuned Flan-T5 generator, a 5-strategy decoding comparison, and a sentence-level
   grounding safeguard: every line shown to a resident is checked against the retrieved
   policy text, with an extractive fallback to the policy text itself when verification
   fails.
5. **Integrated waste management assistant** - a single validated entry point that
   accepts either an image or a text description, confidence-based warnings, response
   caching, a resident feedback loop that exports corrected examples for retraining,
   and an 11/11 edge-case test suite (corrupt files, missing paths, empty input, and more).

## Data

- **RealWaste images** (not bundled): 4,752 images across 9 categories (Cardboard, Food
  Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation).
  Download the dataset and place the class folders under `realwaste/realwaste-main/RealWaste/`
  alongside the notebook before running Part 1.
- **`waste_descriptions.csv`**: place in the same directory as the notebook.
- **`waste_policy_documents.json`**: place in the same directory as the notebook. Part 4
  depends on this file; the notebook will raise a clear error if it is missing.

## Setup

The CNN is the only TensorFlow/Keras component; the text classifier and the RAG
generator both run on PyTorch.

```bash
pip install tensorflow torch transformers sentence-transformers scikit-learn \
    pandas numpy matplotlib pillow
```

No GPU is required. On CPU, expect the CNN architecture sweep and fine-tune to be the
slowest steps (tens of minutes total); the DistilBERT and Flan-T5 fine-tunes each take a
few minutes.

## Running the notebook

**Always start from a clean kernel: Kernel > Restart and Run All.** Do not resume a
previous kernel session or run cells out of order -- several later cells depend on
objects (`rag_index`, `rag_chunks`, fine-tuned model weights) that must be built fresh
in the same session that uses them. A partial or resumed run can silently reuse stale
objects from an earlier session and produce misleading output without raising an error.

Each expensive step (the CNN head sweep, the DistilBERT fine-tune, and the Flan-T5
fine-tune) checks for an existing checkpoint first and skips training if one is found.
To force a clean retrain, delete the corresponding folder under `models/` first.

### A built-in self-check for Part 4

Section 4.3 (`retrieve()`) includes a self-diagnosing demo cell that prints the corpus
size, how many chunks were scored, and how many matched a known test category, before
running the retrieval demo itself. **Read this cell's output before trusting anything
in Part 4 or Part 5.** If it raises a `RuntimeError`, it means the kernel is not running
a fresh build of the retrieval index -- restart the kernel and run every cell again from
the top, in order.

## Known limitations

These are documented in detail in each Part's own "interpretation" cell, with the
actual numbers that support them -- summarised here for convenience:

- The RealWaste category set (9 classes) differs from the 8-class list given in the
  assignment brief; Part 1 documents this and reconciles it against the categories
  actually present in `waste_descriptions.csv`.
- The text classifier's near-perfect accuracy is explained, not just reported: Part 3
  quantifies how much of it is attributable to templated, keyword-separable training
  data and near-duplicate test items, and shows where it breaks down on realistic
  phrasing (Part 5's own standard test cases include a confident misclassification on
  a plain rephrasing of a training example).
- The Flan-T5 generator's fine-tuning improved its fluency (ROUGE-L against the
  reference rose from 0.083 to 0.780) without a matching improvement in grounding
  against the retrieved context, so the safety safeguard in 4.9 falls back to extractive
  policy text for the large majority of evaluated categories. This is treated as a
  feature, not a hidden failure: the assistant is built to never show a resident
  unverified text, and the fallback rate is reported explicitly as the honest measure
  of how often the generator can be trusted unsupervised.
- Whether the category-filtered retrieval path used in production shares the unfiltered
  benchmark's apparent weakness is explicitly flagged as unverified rather than assumed
  either way -- see the self-check described above.
- The policy corpus covers a single jurisdiction with documents dated 2023; every
  generated answer surfaces its source's effective date so a resident (or grader) can
  judge its currency.

## Rubric mapping

Each of the five numbered parts above corresponds directly to one of the five graded
criteria in the assignment rubric (Dataset Exploration and Preparation; Waste Material
Classification with CNN; Waste Description Classification; Recycling Instruction
Generation with RAG; Integrated Waste Management Assistant), each worth 20 points. The
notebook's own introduction cell contains a table mapping each criterion to its exact
section numbers and key evidence, for quick grading reference.
