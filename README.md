# FGPDist-t-SNE

This repository contains the Python code corresponding to the paper "Online Quality Monitoring of Unstructured 3D Point Cloud Data with Improved Manifold Learning Algorithm." In the context of the digital era, where data volume is rapidly increasing and becoming more diverse, products with complex geometric structures generate data with intricate spatiotemporal relationships. This repository aims to address the challenges in quality monitoring by focusing on complex and unstructured 3D point cloud data.

## Overview

In this work, we introduce a new point cloud distance metric termed **Fine-Grained Point Cloud Distance (FGPDist)** and combine it with t-distributed stochastic neighbor embedding (t-SNE) to propose an innovative algorithm model called **FGPDist-t-SNE**. This approach ensures the preservation of detailed information during the conversion of the original 3D point cloud data into a distance matrix, enabling effective dimensionality reduction and feature extraction. We further apply this method for online quality monitoring and anomaly recognition in scenarios where large-scale historical data is unavailable. 

### Application Scenarios
- This work leverages emerging 3D printing technology as a background, validating the proposed algorithm through both simulation experiments and actual data cases.
  
## Code Description

The repository includes the following Python scripts:

- **FGPDist_tsne_sim_pre.py:** Contains the code for running simulation experiments.
- **FGPDist_tsne_realcase_pre.py:** Implements the proposed algorithm using real-world data.
- **Control chart plot.py:** Script for generating control charts used in the analysis.

## Dataset

All the data used in this study is sourced from an open dataset originating from an Open Data Science project between Trumpf GmbH and Politecnico di Milano. The dataset is available at: [Open Data Challenge](https://www.ic.polimi.it/open-data-challenge/).

