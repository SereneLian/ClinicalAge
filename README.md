# ClinicalAge: A Longitudinal EHR-Based Aging Framework

## 🎯 Project Overview

**ClinicalAge** is the official implementation of a transformer-based aging framework derived from longitudinal Electronic Health Records (EHRs). This model predicts a quantitative measure of age—the **ClinicalAge Gap**—from a patient's historical sequence of clinical codes, offering a novel, dynamic, and potentially more accessible biomarker of aging than traditional methods.

The repository provides the full workflow, including data preprocessing, model training, inference, and comprehensive downstream validation.
The UK Biobank PhenoAge formula is based on the method described in https://elifesciences.org/reviewed-preprints/91101.

## Note on data privacy
The original electronic health record (EHR) data used in this study are subject to strict data governance and cannot be shared due to privacy and access restrictions.
In addition, the associated vocabulary files (e.g. 'bert_vocab.pkl', 'year_vocab.pkl') and trained model weights were derived from these restricted data and are therefore not publicly available.
Researchers interested in reproducing this work should obtain appropriate access to relevant EHR datasets (e.g. through UK Biobank or CPRD) and construct their own vocabularies and model weights accordingly.

---

### Prerequisites

You will need Python 3.7+ and access to the necessary protected data files (e.g., UK Biobank and CPRD data) to run the full workflow.
