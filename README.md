# Reconstruction of Masked DNA Sequences

This repository contains the Jupyter Notebook for the prediction phase of the research paper: "Reconstruction of Masked Sequences via Inverse Mapping of Incomplete Information Natural Vectors".

## Overview

This project tackles a key challenge in bioinformatics: reconstructing complete biological sequences from their numerical representations, especially when parts of the original sequence are missing or "masked."

Our approach involves two main steps:
1.  **Feature Extraction**: We introduce a novel method called **Incomplete Information Natural Vectors** to convert masked DNA sequences into fixed-dimensional numerical vectors while preserving crucial positional information.
2.  **Sequence Reconstruction**: We utilize a pre-trained **Long Short-Term Memory (LSTM)** neural network that performs the inverse mapping, successfully reconstructing the original 16-nucleotide sequence from its natural vector with high accuracy.

The model was trained and validated on genomic subsequence datasets from SARS-CoV-2 and HIV-1.

## Jupyter Notebook: `Main.ipynb`

This notebook demonstrates the prediction capabilities of the trained LSTM model. It performs the following steps:

1.  **Loads Data**: Reads a CSV file containing pre-calculated incomplete information natural vectors for a set of masked DNA subsequences.
2.  **Loads Model**: Loads the pre-trained LSTM model from the `saved_model/` directory.
3.  **Predicts Sequences**: Uses the model to predict the original 16-nucleotide sequences from their vector representations.
4.  **Evaluates Accuracy**: Compares the reconstructed sequences to the original sequences to calculate the prediction accuracy.
5.  **Saves Results**: Outputs the predictions and accuracy metrics to a new CSV file.

## How to Run

### Prerequisites

- Python 3.10.15
- Jupyter Notebook or JupyterLab
- Required libraries: `pandas`, `numpy`, `tensorflow`

You can install the dependencies using pip:
```bash
pip install pandas numpy tensorflow
