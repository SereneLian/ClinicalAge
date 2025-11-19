# ClinicalAge: A Longitudinal EHR-Based Aging Framework

## 🎯 Project Overview

**ClinicalAge** is the official implementation of a transformer-based aging framework derived from longitudinal Electronic Health Records (EHRs). This model predicts a quantitative measure of age—the **ClinicalAge Gap**—from a patient's historical sequence of clinical codes, offering a novel, dynamic, and potentially more accessible biomarker of aging than traditional methods.

The repository provides the full workflow for **reproducing the study's results**, including data preprocessing, model training, inference, and comprehensive downstream validation.

The UK Biobank PhenoAge formula is based on the method described in https://elifesciences.org/reviewed-preprints/91101.

*(A high-level diagram illustrating the flow from EHR sequence data to the BERT-like model, predicting the ClinicalAge Gap, and subsequent downstream analyses like biomarker validation and survival modeling)*

## ✨ Key Features & Analyses

The ClinicalAge framework enables several critical downstream analyses, confirming the model's validity and utility:

* **Biomarker Validation:** Association and validation analyses of the ClinicalAge Gap with established biological and functional aging biomarkers.
* **Disease Association:** Calculation of Hazard Ratios (HR) showing the association between the ClinicalAge Gap and over 30 aging-related diseases in two large cohorts (UK Biobank and CPRD).
* **Survival Modeling:** Cox regression analyses demonstrating the predictive power of the ClinicalAge Gap for all-cause and specific-cause mortality.
* **Aging Trajectory Analysis:** Investigation of how the ClinicalAge Gap trajectory changes over time, identifying key ICD/procedure codes associated with acceleration or stability in aging groups.

---

## 🛠️ Getting Started

### Prerequisites

You will need Python 3.7+ and access to the necessary protected data files (e.g., UK Biobank and CPRD data) to run the full workflow.

### Installation

Clone the repository and install the dependencies.

```bash
git clone [https://github.com/SereneLian/ClinicalAge.git](https://github.com/SereneLian/ClinicalAge.git)
cd ClinicalAge
pip install -r requirements.txt
