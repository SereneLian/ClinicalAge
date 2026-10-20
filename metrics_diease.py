
import pandas as pd
from lifelines import CoxPHFitter

def disease_survival(df, disease, cov=None, gap_col='ClinialAge Gap',
                     categorical_cols=None, penalizer=0.0):
    """Disease Cox model.

    df is already merged with the disease-specific event/time data.
    Uses {disease}_event and time2_{disease} 
    cov=None gives an unadjusted model. Non-imputed df gives complete-case
    No automatic penalty escalation is performed: specify it explicitly.
    """


    cov = list(cov or [])
    event_col, time_col = f'{disease}_event', f'time2_{disease}'
    cols = list(dict.fromkeys([gap_col] + cov + [event_col, time_col]))
    data = df[cols].dropna().copy()
    if data.empty:
        raise ValueError(f'{disease}: no complete cases for the selected model.')
    n, events = len(data), int(data[event_col].sum())
    cats = [c for c in (categorical_cols or []) if c in cov]
    if cats:
        data = pd.get_dummies(data, columns=cats, drop_first=True, dtype=float)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data, duration_col=time_col, event_col=event_col)
    result = cph.summary[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']
    ].rename(columns={
        'exp(coef)': 'HR', 'exp(coef) lower 95%': 'CI Lower',
        'exp(coef) upper 95%': 'CI Upper', 'p': 'P-value',
    }).copy()
    result['N'] = n
    result['Events'] = events
    result['Penalizer'] = penalizer
    result['c-index'] = cph.concordance_index_  # Apparent (fitting-sample) C-index.
    return result


cov_age_sex = ['Age', 'Sex']
cov_full = ['Age', 'Sex', 'BMI', 'Systolic_Blood_Pressure',
            'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD', 'ethnicity']


from pathlib import Path
import pandas as pd

disease_dir = '' # data with age, cov and disease label
base_df = ''

rows = []
for file in sorted(Path(disease_dir).glob('*_event_death_censored.csv')):
    disease = file.name[:-len('_event_death_censored.csv')]
    endpoint = pd.read_csv(file).rename(columns={'patid': 'eid'})
    endpoint = endpoint[['eid', f'{disease}_event', f'time2_{disease}']] # processed label df
    merged = endpoint.merge(base_df, on='eid', how='inner')
    result = disease_survival(merged, disease, cov=cov_full, penalizer=0.0)
    row = result.loc['ClinialAge Gap'].copy()
    row['Disease'] = disease
    rows.append(row)
disease_results = pd.DataFrame(rows).reset_index(drop=True)


# merged_imputed = pd.read_csv('') # data with MICE imputed cov and disease label
# merged_non_imputed = pd.read_csv('') # data with non missing cov and disease label
# cprd_merged_imputed = pd.read_csv('')
# ukb_main = disease_survival(merged_imputed, disease, cov=cov_full)
# ukb_complete = disease_survival(merged_non_imputed, disease, cov=cov_full)
# ukb_age_sex = disease_survival(merged_imputed, disease, cov=cov_age_sex)
# cprd_main = disease_survival(cprd_merged_imputed, disease, cov=cov_full, penalizer=0.05)
