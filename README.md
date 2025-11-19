# ClinicalAge
Official code for ClinicalAge: A framework for understanding ageing trajectories from longitudinal electronic health records.

This repository provides the official implementation of **ClinicalAge**, a transformer-based ageing framework derived from longitudinal Electronic Health Records (EHRs). It includes the full workflow used in the study: data processing, EHR sequence generation, model training and inference, and all downstream analyses (biomarker validation, disease associations, survival modelling, and ageing-trajectory analyses). The code is organised to enable transparent reproduction of the results reported in the manuscript and to support future extensions of the ClinicalAge framework.

## Overview

1. `data_process`: UK Biobank data processing and PhenoAge calculation  
2. `ehr_generation`: generation of UK Biobank EHR sequences  
3. `run_ehr_age_train`: training scripts for the ClinicalAge model  
4. `run_ehr_age_inference`: inference on validation datasets / example implementation in clinical settings  Code, including model building and analysis, has been uploaded to GitHub:  (https://github.com/SereneLian/ClinicalAge). Code, including model building and analysis, has been uploaded to GitHub:  (https://github.com/SereneLian/ClinicalAge). 
5. `metrics_age_biomarker`, `metrics_disease`, `metrics_survival`: association and validation analyses of ClinicalAge with biomarkers, functional measures, 31 ageing-related diseases, and survival  
6. `visualise_biomarker`: scripts for biomarker association visualisation (e.g. forest plots)  
7. `trajectory_analyses`, `trajectory_importance`: CPRD trajectory analyses and identification of key ICD codes for different ClinicalAge groups
