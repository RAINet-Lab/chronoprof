# CHRONOPROF: Profiling Time Series Forecasters and Classifiers in Mobile Networks with Explainable AI

This repository contains the artifacts from the study **"CHRONOPROF: Profiling Time Series Forecasters and Classifiers in Mobile Networks with Explainable AI."** We plan to release the artifacts publicly once the paper is accepted.

## Overview

Next-generation mobile networks will increasingly rely on **Artificial Intelligence (AI)** and **Machine Learning (ML)** for automation, orchestration, and effective network management. This involves performing **classification and regression tasks on time-series data**. However, existing AI/ML models are often inherently complex and difficult to interpret. This lack of transparency represents a significant barrier to their deployment in production networks.

CHRONOPROF addresses this limitation by proposing a **new Explainable AI (XAI) tool** specifically designed for time-series classification and regression tasks . Unlike most existing XAI techniques, which were designed for computer vision and natural language processing, CHRONOPROF is **tailored to the unique characteristics of time-series data**.

## The Explainability Problem in Time Series

AI/ML techniques such as DLinear, PatchTST, TSMixer, and MultiRocket have demonstrated significant improvements in accuracy for time-series forecasting and classification across various datasets. However, these models are often **"black boxes"**, making it difficult for network operators to debug and resolve issues . Moreover, they have been shown to be vulnerable to adversarial attacks. The need for explainability is crucial for these models to be deployed effectively in production networks.

Although prominent XAI techniques such as LIME, SHAP, LRP, and DeepLIFT have been adapted for time-series, they often fail to provide explanations that go beyond individual observations. They also do not capture the temporal dynamics of how models adapt to changes in input over time. Furthermore, these techniques are implicitly influenced by the magnitude of input features.

## What is CHRONOPROF?

**CHRONOPROF is a new XAI technique** based on existing XAI methods, specifically SHAP. Its goal is to improve the quality of explanations for multivariate time-series forecasting and classification tasks in mobile networks.

The core principle behind CHRONOPROF is to **isolate the implicit influence of input values** and focus solely on the model’s decision-making process. It does this by generating a **linearized version of the original model** for different observations.

### Key Approach: Virtual Weights

CHRONOPROF leverages the interpretability of linear models to explain more complex DNN models. It generates a set of **virtual weights**, one for each feature, derived from SHAP values. These virtual weights function similarly to coefficients in a linear model.

The key formulation is that virtual weights are computed from SHAP values ($L_i$) and input values ($x_i$) as $v_i = L_i / x_i$. This isolates the influence of feature magnitude and **focuses on the model's sensitivity to each feature change**. Analyzing the evolution of these virtual weights over time provides insights into how the local linear representation of the model adapts to improve accuracy over extended periods.

## Evaluation Setup 
We conducted an extensive evaluation using **real mobile traffic data** for forecasting and classification tasks.

*   **Datasets:**
    *   **D1 (CR):** Traffic volume measurements from a 4G production network in a European metropolitan area. Fine-grained data (10 min) for forecasting and hourly data (24 values/sequence) for classification. Disclaimer: We can not make this dataset public. 
    *   **D2 (R):** Estimated number of active users connected to a production base station, with millisecond-level data aggregated to a 10-minute granularity.
*   **Models:**
    *   **Forecasting:** PatchTST (transformer-based), TSMixer (MLP-based), DLinear (linear).
    *   **Classification:** PatchTST, TSMixer, "Linear" (simple linear layer as baseline), MultiRocket.


## Running the Code

To run the code for forecasting or classification, follow these steps:

### 1. Set up the Environment

First, create a virtual environment using the provided `env.yml` file. This file includes all the dependencies required to run the code.

1. Clone the repository or navigate to the folder where the repository is stored.
2. Create the environment using the following command:

   ```
   conda env create -f env.yml
   ```

Activate the environment:

```
Copy
conda activate <environment_name>
```
Replace ```<environment_name>``` with the name of the environment specified in ```env.yml```.

### 2. Run the Code
After setting up the environment, you can proceed with running the code for either forecasting or classification tasks.

For Forecasting:
To run the forecasting code, use the provided script ```run_mv.sh```. This script is designed to perform time-series forecasting on the data.

Execute the following command:

   ```
   sh run_mv.sh
   ```
For Classification:
To run the classification code, use the provided script ```run_tsc_chinatown.sh```. This script is tailored for time-series classification tasks. Note that the EUMA dataset used for this study is not publicly available. For demonstration purposes, we include the Chinatown dataset as an example.

Execute the following command:

   ```
   sh run_tsc_chinatown.sh
   ```


