# MR-JEPA
A Joint Embedding Predictive Architecture (JEPA) for Mathemetical Reasoning tasks 
# MR-JEPA: Joint Embedding Predictive Architecture for Mathematical Reasoning

## Overview

MR-JEPA is a research project that explores the application of a Joint Embedding Predictive Architecture (JEPA) to mathematical reasoning tasks. Instead of directly generating answers in an autoregressive manner, the model learns latent representations of mathematical problems and predicts future semantic embeddings, allowing reasoning to occur in a structured latent space before decoding into natural language.

The project investigates whether latent predictive learning can improve reasoning capability while reducing reliance on traditional token-by-token generation.

---

## Motivation

Large Language Models have demonstrated impressive mathematical reasoning abilities but often depend heavily on autoregressive decoding. JEPA offers an alternative learning paradigm by predicting latent representations rather than raw outputs.

This project aims to explore whether latent predictive learning can provide a more robust representation for solving mathematical reasoning problems.

---

## Features

* JEPA-based latent representation learning
* Encoder–Predictor–Decoder architecture
* End-to-end mathematical reasoning pipeline
* Baseline evaluation framework
* Training and inference scripts
* Modular implementation for experimentation
* Support for GSM8K formatted datasets

---

## Repository Structure

```text
MR-JEPA/
│
├── Encoders.py                  # Encoder network
├── predictor.py                 # Latent predictor
├── decoder.py                   # Decoder network
├── train.py                     # JEPA training
├── train_decoder.py             # Decoder training
├── test.py                      # Model evaluation
├── test_pipeline.py             # End-to-end testing
├── GSM8K_parser.py              # Dataset preprocessing
├── BaselineEvaluator.py         # Baseline comparisons
│
├── jepa_architecture_flowchart.tsx
├── jepa_decoder_architecture_detail.svg
├── rjepa_overall_architecture.svg
│
├── JEPA for Mathematical Reasoning- A Latent Predictive Approach - final.pdf
├── project_report_midterm.pdf
└── README.md
```

---

## Architecture

<img width="746" height="591" alt="Screenshot 2026-03-27 at 3 23 50 AM" src="https://github.com/user-attachments/assets/1a837351-038b-4344-a345-d426cfdd4a5c" />


The model consists of three primary components:

### Encoder

The encoder converts mathematical problems into compact latent representations that capture semantic information.

### Predictor

The predictor learns to estimate future latent embeddings directly within the embedding space, avoiding token-level prediction during representation learning.

### Decoder

The decoder converts predicted latent embeddings into natural language solutions.

The overall workflow is:

```
Mathematical Problem
        │
        ▼
     Encoder
        │
        ▼
Latent Representation
        │
        ▼
    Predictor
        │
        ▼
Predicted Latent
        │
        ▼
     Decoder
        │
        ▼
Generated Solution
```

---

## Model Pipeline

1. Encode mathematical problems into latent embeddings.
2. Predict future latent representations using the JEPA predictor.
3. Decode predicted embeddings into complete mathematical solutions.
4. Evaluate generated outputs against reference answers.

---

## Dataset

The implementation is designed for the GSM8K mathematical reasoning benchmark.

The dataset is **not included** in this repository due to licensing and repository size considerations.

After obtaining the dataset, place the required files in the project directory before training.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/aurgas/MR-JEPA.git
cd MR-JEPA
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train the JEPA model:

```bash
python train.py
```

Train the decoder:

```bash
python train_decoder.py
```

---

## Evaluation

Evaluate the model:

```bash
python test.py
```

Run the complete inference pipeline:

```bash
python test_pipeline.py
```

---

## Research Objectives

* Investigate JEPA for mathematical reasoning
* Learn meaningful latent representations of mathematical problems
* Compare latent predictive learning with conventional approaches
* Analyse the effectiveness of JEPA on reasoning tasks

---

## Future Work

* Scaling to larger mathematical datasets
* Improved decoder architectures
* Transformer-based latent predictors
* Multi-step latent reasoning
* Evaluation on additional mathematical benchmarks

---

## Repository Contents

This repository includes:

* Source code
* Model architecture diagrams
* Research report
* Project documentation

Datasets, checkpoints and generated outputs are intentionally excluded.

---

## Author

**Poulam Saha**

B.Tech Information Technology

Research interests include Machine Learning, Representation Learning, Computer Vision and Large Language Models.

---

## Licence

This project is intended for research and educational purposes.
