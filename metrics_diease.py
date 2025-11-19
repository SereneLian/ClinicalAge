# %%
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm

# %%
path = '' # prediction result path
ehr_age_all = pd.read_csv(path+'age_predictions.csv')
ehr_age_all.rename(columns={'patid' :'eid','predicted':'ClinialAge Gap', 'baseline_age':'Age'}, inplace=True)
cov = cov = pd.read_csv('data/ukb_cov_imputed.csv')
ehr_age_map_all = ehr_age_all.merge(cov, on='eid', how='left')
save_path = 'results/'+path.split('/')[-2] +'/metrics/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# %%
from lifelines import CoxPHFitter
import pandas as pd
from lifelines import CoxPHFitter



def hr_regression(ints_cols,
                  ehr_age_map_all,
                  category_to_dummy=None,
                  death_label='event',
                  time_label='time2death',
                  penalizer=0):
    """
    Fit a Cox regression model, optionally one-hot-encoding specified categorical columns.

    Parameters
    ----------
    ints_cols : list of str
        Names of the numeric covariate columns.
    ehr_age_map_all : pd.DataFrame
        Input DataFrame containing covariates plus event & time columns.
    category_to_dummy : str or list of str, optional
        Column(s) to be one-hot-encoded (reference level dropped).
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
        plus the model’s concordance index.
    """
    # 1) Copy and prepare integer list
    df = ehr_age_map_all.copy()
    ints = list(ints_cols)

    # 2) One-hot encode any specified categorical columns
    if category_to_dummy is not None:
        cats = [category_to_dummy] if isinstance(category_to_dummy, str) else list(category_to_dummy)
        # cast to int if needed
        for c in cats:
            df[c] = df[c].astype(int)
        # create dummies, drop reference
        df = pd.get_dummies(df, columns=cats, drop_first=True)
        # replace original with dummy columns in ints
        dummy_cols = [col for col in df.columns
                      for orig in cats
                      if col.startswith(f"{orig}_")]
        ints = [col for col in ints if col not in cats] + dummy_cols

    # 3) Subset + drop missing
    df_pred = df[ints + [death_label, time_label]].dropna(subset=ints + [death_label, time_label])

    # 4) Fit Cox model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_pred, duration_col=time_label, event_col=death_label)

    # 5) Extract and format summary
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


def hr_quantile(quantile_col, ints_cols,ehr_age_map_all, death_label='event', time_label='time2death', penalizer=0):
    df_pred = ehr_age_map_all[ints_cols + [death_label, time_label]]    
    df_pred['ClinialAge_quartile'] = pd.qcut(df_pred[quantile_col], q=4, labels=False,duplicates='drop')
        
    df_pred = pd.get_dummies(df_pred, columns=['ClinialAge_quartile'], drop_first=True)
    cph = CoxPHFitter(penalizer=penalizer)
    formula_str = "ClinialAge_quartile_1 + ClinialAge_quartile_2 + ClinialAge_quartile_3"
    # check if additional columns are present
    ints_cols.remove(quantile_col)
    if len(ints_cols) > 0:
        additional_formula = " + ".join(ints_cols)
        formula_str += " + " + additional_formula
    cph.fit(df_pred, duration_col=time_label, event_col=death_label, formula=formula_str)
    # Extract hazard ratios and their confidence intervals
    summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%','z' ,'p']]
    # Round the other columns as desired, but handle the p-values separately
    summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']] = summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']].round(3)
    # Format the p-values in scientific notation to show the specific value
    summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")
    # Get the model-level c-index and round it to 3 decimals
    c_index = round(cph.concordance_index_, 3)
    summary_df['c-index'] = c_index
    return summary_df.loc[summary_df.index.str.contains('ClinialAge_quartile')]

# %%
ukb_disease = pd.read_csv('results/disease_summary_ukb.csv')
disase_list = os.listdir('data/disease_ukb/')
print(disase_list)
disase_list_name = [x[:-10] for x in disase_list]
print(disase_list_name)

# %% [markdown]
# # All Cov Results

# # %%
cov_features =['ClinialAge Gap','Age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 
               'LDL_Cholesterol', 'Smoking','Alcohol', 'IMD', 'ethnicity']
regression_result = []
for dd in disase_list:
    dd_name = dd[:-10]
    dd_df = pd.read_csv('data/disease_ukb/'+dd)
    dd_df.rename(columns={'patid':'eid'}, inplace=True)
    dd_df_analyses = dd_df.merge(ehr_age_map_all, on='eid', how='inner')
    summary_data = hr_regression(cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
    summary = list(summary_data.values[0])
    # # summary_df = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
    regression_result.append([dd_name, *summary])


# %%
regression_df = pd.DataFrame(regression_result, columns=['Disease', 'HR', 'HR_lower', 'HR_upper', 'z', 'p', 'c-index'])
regression_df.to_csv(save_path+'UKB_disease_regression_allcov.csv', index=False)

# # %%
cov_features =['ClinialAge Gap','Age','Sex']
regression_result = []
for dd in disase_list:
    dd_name = dd[:-10]
    dd_df = pd.read_csv('data/disease_ukb/'+dd)
    dd_df.rename(columns={'patid':'eid'}, inplace=True)
    dd_df_analyses = dd_df.merge(ehr_age_map_all, on='eid', how='inner')
    summary_data = hr_regression(cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
    summary = list(summary_data.values[0])
    # # summary_df = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
    regression_result.append([dd_name, *summary])


# %%
regression_df = pd.DataFrame(regression_result, columns=['Disease', 'HR', 'HR_lower', 'HR_upper', 
                                                         'z', 'p', 'c-index'])
regression_df.to_csv(save_path+'UKB_disease_regression_agesex.csv', index=False)

# %% [markdown]
# # Quaantile

# %%
# quantile_result = []
# for dd in disase_list:
#     cov_features =['ClinialAge Gap','Age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 
#                'LDL_Cholesterol', 'Smoking','Alcohol', 'IMD', 'ethnicity']
#     dd_name = dd[:-10]
#     # print('================')
#     print(dd_name)
#     dd_df = pd.read_csv('data/disease_ukb/'+dd)
#     print(dd_df.columns)
#     dd_df.rename(columns={'patid':'eid'}, inplace=True)
#     ehr_age_map_all_use = ehr_age_map_all.copy()
#     dd_df_analyses = dd_df.merge(ehr_age_map_all_use, on='eid', how='inner')
#     summary_data = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
#     quantile_result.append([dd_name, *list(summary_data.values[0])])
#     quantile_result.append([dd_name, *list(summary_data.values[1])])
#     quantile_result.append([dd_name, *list(summary_data.values[2])])



# # %%
# qunatile_df = pd.DataFrame(quantile_result, columns=['Disease', 'HR', 'HR_lower', 'HR_upper', 
#                                                          'z', 'p', 'c-index'])
# qunatile_df.to_csv(save_path+'UKB_disease_quantile_allcov.csv', index=False)

# # %%
# quantile_result = []
# for dd in disase_list:
#     cov_features =['ClinialAge Gap','Age','Sex']
#     dd_name = dd[:-10]
#     # print('================')
#     print(dd_name)
#     dd_df = pd.read_csv('data/disease_ukb/'+dd)
#     print(dd_df.columns)
#     dd_df.rename(columns={'patid':'eid'}, inplace=True)
#     ehr_age_map_all_use = ehr_age_map_all.copy()
#     dd_df_analyses = dd_df.merge(ehr_age_map_all_use, on='eid', how='inner')
#     summary_data = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
#     quantile_result.append([dd_name, *list(summary_data.values[0])])
#     quantile_result.append([dd_name, *list(summary_data.values[1])])
#     quantile_result.append([dd_name, *list(summary_data.values[2])])
# # %%
# qunatile_df = pd.DataFrame(quantile_result, columns=['Disease', 'HR', 'HR_lower', 'HR_upper', 
#                                                          'z', 'p', 'c-index'])
# qunatile_df.to_csv(save_path+'/UKB_disease_quantile_agesex.csv', index=False)
# qunatile_df

# %%

# quantile_result = []

# for dd in disase_list:
#     cov_features =['ClinialAge Gap','phenoage']

#     dd_name = dd[:-10]
#     # print('================')
#     print(dd_name)
#     dd_df = pd.read_csv('data/disease_ukb/'+dd)
#     print(dd_df.columns)
#     dd_df.rename(columns={'patid':'eid'}, inplace=True)
#     ehr_age_map_all_use = ehr_age_map_all.copy()
#     dd_df_analyses = dd_df.merge(ehr_age_map_all_use, on='eid', how='inner')
#     summary_data = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
#     quantile_result.append([dd_name, *list(summary_data.values[0])])
#     quantile_result.append([dd_name, *list(summary_data.values[1])])
#     quantile_result.append([dd_name, *list(summary_data.values[2])])
# qunatile_df = pd.DataFrame(quantile_result, columns=['Disease', 'HR', 'HR_lower', 'HR_upper', 
#                                                          'z', 'p', 'c-index'])
# qunatile_df.to_csv(save_path+'/metrics/UKB_disease_quantile_phenoage.csv', index=False)
# qunatile_df


# %% [markdown]
# # CPRD

# %%
cprd_disease = pd.read_csv('results/disease_summary_cprd.csv')
disase_list = os.listdir('data/disease_cprd/')
print(disase_list)
disase_list_name = [x[:-10] for x in disase_list]
print(disase_list_name)

# %%
# path = '/home/shared/jlian/EHR_image/saved_model_ehr_age/CPRD_randinit_res_exc_age_exc_seg_exc_concat_min_visit1/age_predictions.csv'
path = '/home/shared/jlian/EHR_image/saved_model_ehr_age/CPRD_randinit_bert_small_res_exc_age_exc_seg_exc_concat_time_min_visit1/age_predictions.csv'
cprd_age = pd.read_csv(path)
# cprd_death = pd.read_csv('data/cprd_death_withreason.csv')
# cprd_age_all = cprd_death.merge(cprd_age, on='patid', how='inner')
cov = pd.read_csv('data/cprd_cov_imputed.csv').drop(columns=['baseline_age'])
cov.rename(columns={'eid': 'patid'}, inplace=True)
cprd_age_all = cprd_age.merge(cov, on='patid', how='left')
cprd_age_all.rename(columns={'predicted':'ClinialAge Gap', 'baseline_age':'Age', 'gender':'Sex'}, inplace=True)

# %% [markdown]
# 

# %%
# cov_features =['ClinialAge Gap','Age','Sex']
# regression_result = []
# for dd in disase_list:
#     print(dd)
#     dd_name = dd[:-10]
#     dd_df = pd.read_csv('data/disease_cprd/'+dd)
#     # dd_df.rename(columns={'patid':'eid'}, inplace=True)
#     dd_df_analyses = dd_df.merge(cprd_age_all, on='patid', how='inner')
#     summary_data = hr_regression(cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
#     summary = list(summary_data.values[0])
#     # # summary_df = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name)
#     regression_result.append([dd_name, *summary])
# qunatile_df = pd.DataFrame(regression_result, columns=['Disease', 'HR', 'HR_lower', 'HR_upper', 
#                                                          'z', 'p', 'c-index'])
# qunatile_df.to_csv(save_path+'/cprd_disease_regression_agesex.csv', index=False)
# qunatile_df

# %%
from lifelines.exceptions import ConvergenceError

cov_features =['ClinialAge Gap','Age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 
               'LDL_Cholesterol', 'Smoking','IMD', 'ethnicity']
regression_result = []
for dd in disase_list:
    dd_name = dd[:-10]
    dd_df = pd.read_csv(f'data/disease_cprd/{dd}')
    dd_df_analyses = dd_df.merge(cprd_age_all, on='patid', how='inner')
    dd_df_analyses.dropna(subset=cov_features, inplace=True)
    
    # start with a small penalizer
    pen = 0.05
    max_pen = 1.0
    while True:
        try:
            summary_data = hr_regression(
                cov_features,
                dd_df_analyses,
                death_label = f'{dd_name}_event',
                time_label  = f'time2_{dd_name}',
                penalizer   = pen
            )
            # if we get here, it converged
            break
        except ConvergenceError:
            # bump penalizer and retry
            old = pen
            pen = min(pen * 2, max_pen)
            print(f"{dd_name}: convergence failed at pen={old:.3f}, retrying with pen={pen:.3f}")
            if pen >= max_pen:
                # give up after reaching max_pen
                raise RuntimeError(f"{dd_name}: failed to converge even with penalizer={pen}")
    
    summary = list(summary_data.values[0])
    regression_result.append([dd_name, *summary])

# then save your results as before
qunatile_df = pd.DataFrame(
    regression_result,
    columns=['Disease','HR','HR_lower','HR_upper','z','p','c-index']
)
qunatile_df.to_csv(save_path+'/cprd_disease_regression_allcov.csv', index=False)

# %%
# quantile_result = []
# for dd in disase_list:
#     cov_features =['ClinialAge Gap','Age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 
#                'LDL_Cholesterol', 'Smoking','Alcohol', 'IMD', 'ethnicity']
#     dd_name = dd[:-10]
#     # print('================')
#     print(dd_name)
#     dd_df = pd.read_csv('data/disease_cprd/'+dd)
#     dd_df.rename(columns={'patid':'eid'}, inplace=True)
#     ehr_age_map_all_use = ehr_age_map_all.copy()
#     dd_df_analyses = dd_df.merge(ehr_age_map_all_use, on='eid', how='inner')
#     summary_data = hr_quantile('ClinialAge Gap', cov_features, dd_df_analyses, dd_name +'_event', 'time2_'+dd_name, penalizer=0.05)
#     quantile_result.append([dd_name, *list(summary_data.values[0])])
#     quantile_result.append([dd_name, *list(summary_data.values[1])])
#     quantile_result.append([dd_name, *list(summary_data.values[2])])



# %%



