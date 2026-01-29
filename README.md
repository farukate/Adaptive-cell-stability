# Adaptive Cell-Based Stability

This repository provides the reference implementation for the experiments reported in the manuscript:

**Stabilizing Neighborhood-Based Inference via Adaptive Cell-Based Spherical Representation**

The code is released to support transparency and reproducibility of the experimental results.

---

## Overview

Neighborhood-based inference methods such as k-nearest neighbors (KNN) are highly sensitive to small input perturbations, particularly in high-dimensional spaces.  
This project investigates decision stability as a representational property and introduces a cell-based spherical abstraction with adaptive refinement to stabilize neighborhood inference without modifying the underlying decision rule.

The repository includes point-based baselines, uniform cell-based representations, and an adaptive refinement mechanism evaluated under identical experimental conditions.

---

## Repository Structure

adaptive-cell-stability/
│
├── experiment_v1_point_baselines.py
├── experiment_v2_cell_uniform.py
├── experiment_v3_adaptive_refinement.py
├── requirements.txt
├── LICENSE
└── README.md


- **experiment_v1_point_baselines.py**  
  Point-based KNN baselines using Euclidean and angular (cosine) distance.

- **experiment_v2_cell_uniform.py**  
  Uniform cell-based spherical representation using k-means clustering (k=50), without refinement.

- **experiment_v3_adaptive_refinement.py**  
  Cell-based representation with ambiguity-triggered adaptive refinement.

---

## Dataset

Experiments are conducted using the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository.

The dataset is **not included** in this repository and must be obtained separately:

- UCI ML Repository:  
  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

After downloading, place the file as:

data.csv


in the root directory of the repository.

---

## Requirements

The code has been tested with Python 3.10+.  
Required packages are listed in `requirements.txt`:

numpy
pandas
scikit-learn
matplotlib

Install dependencies via:

```bash
pip install -r requirements.txt


Usage

Each experiment script can be executed independently:

python experiment_v1_point_baselines.py
python experiment_v2_cell_uniform.py
python experiment_v3_adaptive_refinement.py


All scripts use fixed random seeds and identical experimental protocols to ensure reproducibility.

License

This project is released under the MIT License.

Citation

If you use this code in academic work, please cite the associated manuscript:

Stabilizing Neighborhood-Based Inference via Adaptive Cell-Based Spherical Representation


A formal bibliographic entry will be added upon publication.










