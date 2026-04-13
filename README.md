# CSE382 Major Project: Image Classifiers

This repository contains the MNIST classification project for CAIE CSE382 Spring 2026.

## Project Scope

- **Phase 1:** Binary classification on MNIST, for example `0 vs 1`
- **Phase 2:** Multiclass classification on all `10` digits
- **Goal:** Build a full machine learning pipeline with experiments, metrics, and report-ready analysis

## Current Structure

- [src/features](src/features) for reusable preprocessing and feature functions
- [src/models](src/models) for reusable model training and hyperparameter selection functions
- [src/tests](src/tests) for automated tests and evaluation utilities
- [src/analysis](src/analysis) for visualization and experiment-analysis helpers
- [src/pipelines](src/pipelines) for notebook-driven experiment orchestration
- [src/tests](src/tests) for automated tests
- [data/mnist.npz](data/mnist.npz) for the MNIST data file
- [reports](reports) for outputs, tables, and figures

## What Is Implemented Now

- MNIST exploration notebooks
- Feature preparation notebooks
- Phase 1 KNN pipeline split into reusable modules
- Clear separation of concerns:
  - feature engineering in [src/features](src/features)
  - model-only logic in [src/models](src/models)
  - evaluation/output logic in [src/evaluation](src/evaluation)
  - plotting/analysis in [src/analysis](src/analysis)
- `.env` support for seed, class selection, split ratios, and feature settings
- Saved experiment outputs including metrics, confusion matrices, validation tuning tables, and KNN summary files

## Notebooks

- [src/features/mnist_dataset_exploration.ipynb](src/features/mnist_dataset_exploration.ipynb): inspect the dataset, labels, and sample images
- [src/features/mnist_feature_prep_workflow.ipynb](src/features/mnist_feature_prep_workflow.ipynb): see normalization, splitting, flattening, PCA, and HOG
- [src/pipelines/phase1_knn_pipeline.ipynb](src/pipelines/phase1_knn_pipeline.ipynb): notebook orchestration for the binary KNN experiment

## Python Modules

- [src/features/mnist_features.py](src/features/mnist_features.py): loading, splitting, normalization, and feature extraction helpers
- [src/models/knn_model.py](src/models/knn_model.py): KNN training/tuning and feature-mode experiment execution
- [src/tests/knn_evaluation.py](src/tests/knn_evaluation.py): test metrics + artifact saving (`validation_tuning.csv`, `metrics.json`, `confusion_matrix.csv`)
- [src/analysis/knn_analysis.py](src/analysis/knn_analysis.py): tuning-curve and confusion-matrix plotting helpers
- [src/pipelines/phase1_knn.py](src/pipelines/phase1_knn.py): compatibility wrappers for existing imports and tests

## Environment File

Create a `.env` file in the project root from [\.env.example](.env.example). Supported values include:

- `RANDOM_STATE`
- `DATA_PATH`
- `CLASS_A`
- `CLASS_B`
- `TEST_SIZE`
- `VAL_SIZE`
- `MODELS`
- `K_VALUES`
- `PCA_COMPONENTS`
- `FEATURES`
- `OUTPUT_DIR`

## Quick Start

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Copy [\.env.example](.env.example) to `.env` and adjust values if needed.
3. Open and run [src/features/mnist_dataset_exploration.ipynb](src/features/mnist_dataset_exploration.ipynb) first.
4. Open and run [src/features/mnist_feature_prep_workflow.ipynb](src/features/mnist_feature_prep_workflow.ipynb) next.
5. Run the Phase 1 notebook pipeline:
   - [src/pipelines/phase1_knn_pipeline.ipynb](src/pipelines/phase1_knn_pipeline.ipynb)

## Phase 1 Outputs

The KNN-focused pipeline writes results to the folder configured in `OUTPUT_DIR` (for example [reports/phase1_knn_visual](reports/phase1_knn_visual)).

- `summary_knn.csv`
- `best_knn.json`
- `run_info.json`
- per-feature folders (`flatten/`, `pca/`, and `hog/`) with tuning and confusion matrix files

## Recommended Next Tasks

1. Add Logistic Regression as a second model.
2. Add Linear SVM as a third model.
3. Build a comparison table for all Phase 1 models.
4. Move to Phase 2 after the Phase 1 baseline is stable.

## Milestone Checklist

### Milestone 1

- Binary classification working
- At least 3 algorithms compared
- Metrics table and confusion matrix
- 5–7 page report draft

### Milestone 2

- 10-class system complete
- At least 3 improvement strategies applied
- Final 10–15 page report, presentation, and demo
