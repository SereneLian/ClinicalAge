import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

def survival(df, cov=None, event_col='event', time_col='time2death',
             gap_col='ClinialAge Gap', categorical_cols=None):
    """Cox regression using the supplied, already-preprocessed event/time data.
    cov: adjustment covariates; None or [] gives the unadjusted model.
    categorical_cols: covariates to dummy-code, e.g. ['Smoking', 'ethnicity'].
    Missing rows are dropped only for columns used in this model. Pass a merged
    non-imputed DataFrame for complete-case sensitivity analysis.
    Returns HR, 95% CI, P-value, N and Events for each fitted predictor.
    """
    cov = list(cov or [])
    cols = list(dict.fromkeys([gap_col] + cov + [event_col, time_col]))
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError('No complete cases for the selected model.')
    n, events = len(data), int(data[event_col].sum())

    cats = [c for c in (categorical_cols or []) if c in cov]
    if cats:
        data = pd.get_dummies(data, columns=cats, drop_first=True, dtype=float)

    cph = CoxPHFitter()
    cph.fit(data, duration_col=time_col, event_col=event_col)
    result = cph.summary[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']
    ].rename(columns={
        'exp(coef)': 'HR',
        'exp(coef) lower 95%': 'CI Lower',
        'exp(coef) upper 95%': 'CI Upper',
        'p': 'P-value',
    }).copy()
    result['N'] = n
    result['Events'] = events
    return result


df_imputed = pd.read_csv('') #load MICE imputed df
df_non_imputed = pd.read_csv('') # load non missing cov df
cov1 = []
cov2 = ['PhenoAge']
cov3 = ['Age', 'Sex']
cov4 = [
    'Age', 'Sex', 'BMI', 'Systolic Blood Pressure',
    'HDL Cholesterol', 'LDL Cholesterol',
    'Smoking', 'IMD', 'ethnicity'
]

cov_models = {'M1': cov1, 'M2': cov2, 'M3': cov3, 'M4': cov4}
categorical_cols = ['Sex', 'Smoking', 'ethnicity']

main = {
    name: survival(df_imputed, cov=cov, categorical_cols=categorical_cols)
    for name, cov in cov_models.items()
}

# Complete-case
complete = {
    name: survival(df_non_imputed, cov=cov, categorical_cols=categorical_cols)
    for name, cov in cov_models.items()
}

# cancer = {
#     name: survival(
#         df_non_imputed, cov=cov,
#         event_col='Cancer_Flag',
#         categorical_cols=categorical_cols
#     )
#     for name, cov in cov_models.items()
# }



