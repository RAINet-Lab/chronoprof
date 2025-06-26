# CHRONOPROF: Profiling Time Series Forecasters and Classifiers in Mobile Networks with Explainable AI

[![DOI](https://zenodo.org/badge/980072982.svg)](https://doi.org/10.5281/zenodo.15745345)

-----

This repository contains the artifacts from the study **"CHRONOPROF: Profiling Time Series Forecasters and Classifiers in Mobile Networks with Explainable AI."**

This work will be presented at the IEEE International Symposium on a World of Wireless, Mobile and Multimedia Networks (**WoWMoM**) 2025.

## Overview

Next-generation mobile networks will increasingly rely on **Artificial Intelligence (AI)** and **Machine Learning (ML)** for automation, orchestration, and effective network management. This involves performing **classification and regression tasks on time-series data**. However, existing AI/ML models are often inherently complex and difficult to interpret. This lack of transparency represents a significant barrier to their deployment in production networks.

CHRONOPROF addresses this limitation by proposing a **new Explainable AI (XAI) tool** specifically designed for time-series classification and regression tasks . Unlike most existing XAI techniques, which were designed for computer vision and natural language processing, CHRONOPROF is **tailored to the unique characteristics of time-series data**.

## The Explainability Problem in Time Series

AI/ML techniques such as DLinear, PatchTST, TSMixer, and MultiRocket have demonstrated significant improvements in accuracy for time-series forecasting and classification across various datasets. However, these models are often **"black boxes"**, making it difficult for network operators to debug and resolve issues. Moreover, they have been shown to be vulnerable to adversarial attacks. The need for explainability is crucial for these models to be deployed effectively in production networks.

Although prominent XAI techniques such as LIME, SHAP, LRP, and DeepLIFT have been adapted for time-series, they often fail to provide explanations that go beyond individual observations. They also do not capture the temporal dynamics of how models adapt to changes in input over time.

## What is CHRONOPROF?

**CHRONOPROF is a new XAI technique** based on existing XAI methods, specifically SHAP. Its goal is to improve the quality of explanations for multivariate time-series forecasting and classification tasks in mobile networks.

The core principle behind CHRONOPROF is to **isolate the implicit influence of input values** and focus solely on the model’s decision-making process. It does this by generating a **linearized version of the original model** for each sample.

## Evaluation Setup 
We conducted an extensive evaluation using **real mobile traffic data** for forecasting and classification tasks.

*   **Datasets:**
    *   **EUMA:** Traffic volume measurements from a 4G production network in a European metropolitan area. Fine-grained data (10 min) for forecasting and hourly data (24 values/sequence) for classification. Disclaimer: We can not make this dataset public. 
    *   **[RRC Connected Users](https://git2.networks.imdea.org/wng/madrid-lte-dataset):** Estimated number of active users connected to a production base station, with millisecond-level data aggregated to a 10-minute granularity.
*   **Models:**
    *   **Forecasting:** PatchTST (transformer-based), TSMixer (MLP-based), DLinear (linear).
    *   **Classification:** PatchTST, TSMixer, "Linear" (simple linear layer as baseline), MultiRocket.

## 📥 Download Datasets

To reproduce the experiments in this repository, you will need to manually download the required datasets and place them in the `dataset/` directory:

- **[RRC Connected Users](https://box.networks.imdea.org/s/wxiZamiEXA5aVGx)** 

- **[Chinatown](https://github.com/iwuqing/Time-Series-Classification-based-on-KNN/tree/master/data/Chinatown):** This dataset is a well known public time series classification benchmark.

Once downloaded, please place the datasets inside the `dataset` folder as follows:

```
dataset
├── Chinatown
│ ├── Chinatown_TRAIN.tsv
│ ├── Chinatown_TEST.tsv
│ └── README.md
└── users_allBS.csv
```

## Running the Code

To run the code for forecasting or classification, follow these steps:

### 1. Set up the Environment

First, create a virtual environment using the provided `env.yml` file. This file includes all the dependencies required to run the code.

1. Clone the repository or navigate to the folder where the repository is stored.
2. Create the environment using the following command:

   ```
   conda env create -f env.yml
   ```

3. Activate the environment:

```
conda activate <environment_name>
```
Replace ```<environment_name>``` with the name of the environment specified in ```env.yml```.

### 2. Run the Code
After setting up the environment, you can proceed with running the code for either forecasting or classification tasks.

To run the **forecasting** code example, use the provided script ```run_mv.sh```.

Execute the following command:

   ```
   sh run_mv.sh
   ```

To run the **classification** code example, use the provided script ```run_tsc_chinatown.sh```. Note that the EUMA dataset used for this study is not publicly available. For demonstration purposes, we include the [Chinatown](https://github.com/iwuqing/Time-Series-Classification-based-on-KNN/tree/master/data/Chinatown) dataset as an example.

Execute the following command:

   ```
   sh run_tsc_chinatown.sh
   ```

## Visualization

To visualize the results, please refer to the notebook ```visualization-forecasting.ipynb ```. This notebook provides a detailed guide on how to visualize the XAI outcomes and gain insights from the model behaviour.



