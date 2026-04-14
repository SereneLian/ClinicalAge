# %%
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from lifelines import CoxPHFitter

# %%
path = ''  # prediction result path
ehr_age_all = pd.read_csv(path + 'age_predictions.csv')
ehr_age_all.rename(columns={'patid': 'eid', 'predicted': 'ClinialAge Gap', 'baseline_age': 'Age'}, inplace=True)

# Load covariates
cov = pd.read_csv('data/ukb_cov_imputed.csv')
ehr_age_map_all = ehr_age_all.merge(cov, on='eid', how='left')

# Load death data
ukb_death = pd.read_csv('data/ukb_death.csv')
ukb_death.rename(columns={'patid': 'eid'}, inplace=True)
ehr_age_map_all = ehr_age_map_all.merge(ukb_death, on='eid', how='left')

save_path = 'results/' + path.split('/')[-2] + '/metrics/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# %%
def hr_regression(ints_cols,
                  ehr_age_map_all,
                  death_label='event',
                  time_label='time2death',
                  penalizer=0):
    """
    Fit a Cox regression model with numeric covariates only (Age and Sex).

    Parameters
    ----------
    ints_cols : list of str
        Names of the numeric covariate columns.
    ehr_age_map_all : pd.DataFrame
        Input DataFrame containing covariates plus event & time columns.
    death_label : str
        Name of the event indicator column.
    time_label : str
        Name of the time-to-event column.
    penalizer : float
        L2 penalty for the Cox model.

    Returns
    -------
    pd.DataFrame
        Formatted summary of hazard ratios, 95% CIs, z-scores, p-values,
        plus the model's concordance index.
    """
    df = ehr_age_map_all.copy()
    ints = list(ints_cols)

    # Subset + drop missing
    df_pred = df[ints + [death_label, time_label]].dropna(subset=ints + [death_label, time_label])

    # Fit Cox model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_pred, duration_col=time_label, event_col=death_label)

    # Extract and format summary
    summary_df = cph.summary.loc[ints, [
        'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p'
    ]].copy()

    # Round HRs & z, format p-values
    summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']] = \
        summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']].round(3)
    summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")

    # Add concordance index
    summary_df['c-index'] = round(cph.concordance_index_, 3)

    return summary_df


def hr_quantile(quantile_col, ints_cols, ehr_age_map_all, death_label='event', time_label='time2death', penalizer=0):
    """
    Fit a Cox regression model with ClinicalAge Gap quartiles.
    """
    df_pred = ehr_age_map_all[ints_cols + [death_label, time_label]].copy()
    df_pred = df_pred.dropna()
    df_pred['ClinialAge_quartile'] = pd.qcut(df_pred[quantile_col], q=4, labels=False, duplicates='drop')

    df_pred = pd.get_dummies(df_pred, columns=['ClinialAge_quartile'], drop_first=True)
    cph = CoxPHFitter(penalizer=penalizer)
    formula_str = "ClinialAge_quartile_1 + ClinialAge_quartile_2 + ClinialAge_quartile_3"
    # check if additional columns are present
    ints_cols_copy = list(ints_cols)
    ints_cols_copy.remove(quantile_col)
    if len(ints_cols_copy) > 0:
        additional_formula = " + ".join(ints_cols_copy)
        formula_str += " + " + additional_formula
    cph.fit(df_pred, duration_col=time_label, event_col=death_label, formula=formula_str)
    # Extract hazard ratios and their confidence intervals
    summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']]
    # Round the other columns as desired, but handle the p-values separately
    summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']] = summary_df[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']].round(3)
    # Format the p-values in scientific notation to show the specific value
    summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")
    # Get the model-level c-index and round it to 3 decimals
    c_index = round(cph.concordance_index_, 3)
    summary_df['c-index'] = c_index
    return summary_df.loc[summary_df.index.str.contains('ClinialAge_quartile')]


# Only adjust for Age and Sex to ensure consistent handling
cov_features = ['ClinialAge Gap', 'Age', 'Sex']
# All-cause mortality
summary_allcause = hr_regression(cov_features, ehr_age_map_all, 'event', 'time2death')
summary_allcause.to_csv(save_path + 'UKB_allcause_mortality_agesex.csv')
print("All-cause mortality (Age & Sex adjusted):")
print(summary_allcause)

# # Cause-Specific Mortality - Age and Sex Adjusted

# %%
# Cancer mortality
summary_cancer = hr_regression(cov_features, ehr_age_map_all, 'Cancer_Flag', 'time2death')
summary_cancer.to_csv(save_path + 'UKB_cancer_mortality_agesex.csv')
print("\nCancer mortality (Age & Sex adjusted):")
print(summary_cancer)

# %%
# Cardiovascular mortality
summary_cardio = hr_regression(cov_features, ehr_age_map_all, 'Cardio_Flag', 'time2death')
summary_cardio.to_csv(save_path + 'UKB_cardio_mortality_agesex.csv')
print("\nCardiovascular mortality (Age & Sex adjusted):")
print(summary_cardio)

# %% [markdown]
# # Quartile Analysis - Age and Sex Adjusted

# %%
# Quartile analysis for all-cause mortality
summary_quantile_allcause = hr_quantile('ClinialAge Gap', cov_features, ehr_age_map_all, 'event', 'time2death')
summary_quantile_allcause.to_csv(save_path + 'UKB_allcause_mortality_quantile_agesex.csv')
print("\nAll-cause mortality by ClinicalAge Gap quartiles (Age & Sex adjusted):")
print(summary_quantile_allcause)

# %%
# Quartile analysis for cancer mortality
summary_quantile_cancer = hr_quantile('ClinialAge Gap', cov_features, ehr_age_map_all, 'Cancer_Flag', 'time2death')
summary_quantile_cancer.to_csv(save_path + 'UKB_cancer_mortality_quantile_agesex.csv')
print("\nCancer mortality by ClinicalAge Gap quartiles (Age & Sex adjusted):")
print(summary_quantile_cancer)

# %%
# Quartile analysis for cardiovascular mortality
summary_quantile_cardio = hr_quantile('ClinialAge Gap', cov_features, ehr_age_map_all, 'Cardio_Flag', 'time2death')
summary_quantile_cardio.to_csv(save_path + 'UKB_cardio_mortality_quantile_agesex.csv')
print("\nCardiovascular mortality by ClinicalAge Gap quartiles (Age & Sex adjusted):")
print(summary_quantile_cardio)
