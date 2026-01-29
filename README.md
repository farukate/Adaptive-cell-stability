# Stabilizing Neighborhood-Based Inference via Adaptive Cell-Based Spherical Representation

This repository contains the experimental code and results accompanying the paper:

**"Stabilizing Neighborhood-Based Inference via Adaptive Cell-Based Spherical Representation"**

## Contents

This repository includes:

- Point-based KNN baseline experiments (Euclidean and angular distance)
- Cell-based spherical representation with adaptive refinement
- Cross-validation results stored as CSV files
- Scripts for reproducing all figures reported in the paper

## Reproducibility

All experiments use fixed random seeds and deterministic preprocessing steps.
Running the provided Python scripts reproduces the reported CSV result files,
which are used to generate the figures in the manuscript.

## Requirements

The experiments require the following Python packages:

- numpy
- pandas
- scikit-learn
- matplotlib

