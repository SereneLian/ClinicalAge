# %%
import pandas as pd
import numpy as np
import os
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats


# %%
path = ''
ehr_age_all = pd.read_csv(path+'age_predictions.csv') #
ehr_age_test = pd.read_csv(path+'age_predictions_test.csv')
actual_age = pd.read_csv('data/ukb_death.csv')
actual_age.rename(columns={'eid':'patid'}, inplace=True)
ehr_age_map_all = ehr_age_all.merge(actual_age, on='patid', how='inner')
ehr_age_map_test = ehr_age_test.merge(actual_age, on='patid', how='inner')
ehr_age_map_test['predicted_age'] = ehr_age_map_test['baseline_age'] + ehr_age_map_test['predicted']
ehr_age_map_all['predicted_age'] = ehr_age_map_all['baseline_age'] + ehr_age_map_all['predicted']
print('all data:', ehr_age_map_all.shape)
save_path = 'results/'+path.split('/')[-2]+'/metrics/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

cov_ukb = pd.read_csv('data/ukb_cov_imputed.csv')
cov_cprd = pd.read_csv('data/cprd_cov_imputed.csv').drop(columns=['baseline_age'])
# cov_cprd = pd.read_csv('data/cprd_cov_non_missing.csv').drop(columns=['baseline_age'])
cov_cprd['Sex'].replace({2:0}, inplace=True)

# %%

def bootstrap_ci(data, metric_func, n_bootstraps=1000, alpha=0.05):
    """Compute bootstrap confidence interval for a given metric function."""
    bootstrapped_metrics = []
    n = len(data)
    for _ in range(n_bootstraps):
        sample = data.sample(n, replace=True)
        bootstrapped_metrics.append(metric_func(sample))
    lower_bound = np.percentile(bootstrapped_metrics, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrapped_metrics, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound

def metrics_report(ehr_age_map,dataset, gt_label = 'phenoage',n_bootstraps=1000, alpha=0.05):
    # Define metric functions
    def r2_metric(df):
        return r2_score(df[gt_label], df["predicted_age"])

    def pearson_metric(df):
        return df["predicted_age"].corr(df[gt_label], method="pearson")

    def mae_metric(df):
        return mean_absolute_error(df[gt_label], df["predicted_age"])

    def mse_metric(df):
        return mean_squared_error(df[gt_label], df["predicted_age"])

    def mape_metric(df):
        return np.mean(np.abs((df[gt_label] - df["predicted_age"]) / df[gt_label])) * 100
    
    def pred_diff_metric(df):
        return (df[gt_label] - df["predicted_age"]).mean()

    # Calculate metrics and their confidence intervals
    metrics = {
        "R²": (r2_metric(ehr_age_map), bootstrap_ci(ehr_age_map, r2_metric, n_bootstraps, alpha)),
        "Pearson Correlation": (pearson_metric(ehr_age_map), bootstrap_ci(ehr_age_map, pearson_metric, n_bootstraps, alpha)),
        "Mean Absolute Error": (mae_metric(ehr_age_map), bootstrap_ci(ehr_age_map, mae_metric, n_bootstraps, alpha)),
        "Mean Squared Error": (mse_metric(ehr_age_map), bootstrap_ci(ehr_age_map, mse_metric, n_bootstraps, alpha)),
        "Mean Absolute Percentage Error": (mape_metric(ehr_age_map), bootstrap_ci(ehr_age_map, mape_metric, n_bootstraps, alpha)),
        "Mean Predicted Age Difference": (pred_diff_metric(ehr_age_map), bootstrap_ci(ehr_age_map, pred_diff_metric, n_bootstraps, alpha))
    }

    metric_results = []
    for metric, (value, (ci_lower, ci_upper)) in metrics.items():
        metric_results.append({
            "Metric": metric,
            "Value": value,
            "CI Lower Bound": ci_lower,
            "CI Upper Bound": ci_upper
        })
        # Also print the results if desired
        print(f"{metric}: {value:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
    metrics_df = pd.DataFrame(metric_results)
    metrics_df.to_csv(save_path+f'/metrics_results_{dataset}_{gt_label}.csv', index=False)
    # Group by 'Age_range' and calculate mean 'pred_diff' and 'diff'
    print('Predicted BA vs CA:')
    predicted_by_age = ehr_age_map.groupby('Age_range')['predicted'].mean().reset_index()
    predicted_by_age.to_csv(save_path+ f"/pred_diff_age_range_{dataset}_{gt_label}.csv", index=False)
    print(predicted_by_age)
    print('GT BA vs CA:')
    print(ehr_age_map.groupby('Age_range')['diff'].mean())

# # %%
metrics_report(ehr_age_map_test,'test','phenoage' )
metrics_report(ehr_age_map_test,'test','baseline_age' )
metrics_report(ehr_age_map_all, 'all', 'baseline_age')
metrics_report(ehr_age_map_all, 'all')


# %% [markdown]
# # Survival Analyses

# %%
from lifelines import CoxPHFitter
cov = cov_ukb[['eid', 'Sex', 'BMI', 'Systolic Blood Pressure', 'HDL Cholesterol',
       'LDL Cholesterol', 'Smoking', 'ethnicity', 'Alcohol', 'IMD']]
cov = cov_ukb.rename(columns={'eid':'patid'})
fi_df = pd.read_csv('data/frailty_index_final.csv')
fi_df = fi_df[['eid','frailty index']].rename(columns={'eid':'patid'})
ehr_age_map_all = ehr_age_map_all.merge(cov, on='patid', how='inner')
ehr_age_map_all = ehr_age_map_all.merge(fi_df, on='patid', how='inner')

# %% [markdown]
# ## use predicted age

# %%
def hr_regression(ints_cols, ehr_age_map_all, category_to_dummy=None, death_label='event', penalizer=0):
    # 1) Copy and (later) drop rows with missing predictor data
    df = ehr_age_map_all.copy()

    # 2) One-hot–encode any specified categorical column(s)
    if category_to_dummy is not None:
        df[category_to_dummy] = df[category_to_dummy].astype('int')
        # allow either a single col or list
        cats = [category_to_dummy] if isinstance(category_to_dummy, str) else list(category_to_dummy)
        df = pd.get_dummies(df, columns=cats, drop_first=True)
        # adjust ints_cols: remove originals, add new dummies
        dummy_cols = [c for c in df.columns
                      for orig in cats
                      if c.startswith(f"{orig}_")]
        ints_cols = [c for c in ints_cols if c not in cats] + dummy_cols
    print(f'ints_cols: {ints_cols}')
    # 3) Drop rows with NaNs in any of the predictor columns
    print(f'before: {len(df)} rows')
    df = df.dropna(subset=ints_cols)
    print(f' after: {len(df)} rows')

    # 4) Build dataframe for CoxPHFitter
    df_pred = df[ints_cols + [death_label, 'time2death']]

    # 5) Fit the model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_pred, duration_col='time2death', event_col=death_label)

    # 6) Extract and format results
    summary_df = cph.summary[
        ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']
    ].copy()

    # Round numeric columns
    for col in ['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']:
        summary_df[col] = summary_df[col].round(3)

    # Scientific notation for p-values
    summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")

    # Add model-level c-index
    summary_df['c-index'] = round(cph.concordance_index_, 3)

    return summary_df


# %%
def hr_quantile(quantile_col, ints_cols, ehr_age_map_all, category_to_dummy=None, death_label='event', penalizer=0):
    print(f'before: {len(ehr_age_map_all)}')

    # Ensure we don't modify the original list
    ints_cols = list(ints_cols)

    if category_to_dummy is not None:
            ehr_age_map_all[category_to_dummy] = ehr_age_map_all[category_to_dummy].astype('int')
            # allow either a single col or list
            cats = [category_to_dummy] if isinstance(category_to_dummy, str) else list(category_to_dummy)
            ehr_age_map_all = pd.get_dummies(ehr_age_map_all, columns=cats, drop_first=True)
            # adjust ints_cols: remove originals, add new dummies
            dummy_cols = [c for c in ehr_age_map_all.columns
                        for orig in cats
                        if c.startswith(f"{orig}_")]
            ints_cols = [c for c in ints_cols if c not in cats] + dummy_cols
    print(f'ints_cols: {ints_cols}')

    # Ensure all required columns are present before dropna
    required_cols = list(set(ints_cols + [quantile_col, death_label, 'time2death']))
    ehr_age_map_all = ehr_age_map_all.copy().dropna(subset=required_cols)

    print(f'after dropna: {len(ehr_age_map_all)}')

    # Prepare dataframe
    df_pred = ehr_age_map_all[required_cols].copy()

    # Create quantile-based categorical variable
    df_pred['ClinicalAge_quartile'] = pd.qcut(df_pred[quantile_col], q=4, labels=False, duplicates='drop')

    # One-hot encode quartiles, dropping the reference group
    df_pred = pd.get_dummies(df_pred, columns=['ClinicalAge_quartile'], prefix='Q', drop_first=True)

    # Build formula: include dummy quartile columns and any other covariates
    quartile_cols = [col for col in df_pred.columns if col.startswith('Q_')]
    other_covariates = [col for col in ints_cols if col != quantile_col]
    all_covariates = quartile_cols + other_covariates
    formula_str = " + ".join(all_covariates)

    print("Formula:", formula_str)

    # Fit Cox model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_pred, duration_col='time2death', event_col=death_label, formula=formula_str)

    # Extract and format summary
    summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']]
    summary_df[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']] = summary_df[[
        'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z']].round(3)
    summary_df['p'] = summary_df['p'].apply(lambda x: f"{x:.3e}")
    summary_df['c-index'] = round(cph.concordance_index_, 3)

    return summary_df

category_to_dummy = ['Smoking','ethnicity', "IMD"]

# %%
# # using diff
pred_age = hr_regression(['predicted'],ehr_age_map_all) #clinicalAge
# print(pred_age)
pheno_age = hr_regression(['label'],ehr_age_map_all)#PhenoAge
# print(pheno_age)
fi_age = hr_regression(['frailty index'],ehr_age_map_all)#fi
# print(fi_age)
pred_pheno = hr_regression(['predicted','phenoage'],ehr_age_map_all) # model 2
# print(pred_pheno)
pred_sex_age = hr_regression(['predicted','baseline_age','Sex'],ehr_age_map_all) # model 3
# print(pred_sex_age)
cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
pred_cov = hr_regression(cov_col,ehr_age_map_all, category_to_dummy)
# print(pred_cov)
combined = pd.concat([pred_age, pheno_age, fi_age, pred_pheno, pred_sex_age, pred_cov],
    keys=['pred_age','pheno_age' ,'frailty index', 'pred_pheno', 'pred_sex_age', 'pred_cov'],axis=0)
combined.to_csv(save_path+'UKB_cox_regession_predicted_diff.csv')


# # %%
# pred_age_q = hr_quantile('predicted',['predicted'], ehr_age_map_all)
# phenoage_q = hr_quantile('label',['label'], ehr_age_map_all)
# fi_age_q = hr_quantile('frailty index', ['frailty index'],ehr_age_map_all)
# # print('frailty index')
# pred_pheno_q = hr_quantile('predicted',['predicted','phenoage'],ehr_age_map_all)
# # print(pred_age_q)
# pred_sex_age_q = hr_quantile('predicted',['predicted','baseline_age','Sex'],ehr_age_map_all)
# # print(pred_sex_age_q)
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking','IMD','ethnicity']
# pred_cov_q = hr_quantile('predicted', cov_col,ehr_age_map_all, category_to_dummy)
# # print(pred_cov_q)
# q_combined = pd.concat([pred_age_q,phenoage_q, fi_age_q ,pred_pheno_q, pred_sex_age_q, pred_cov_q],
#     keys=['pred_age_diff', 'phenoage', 'frailty index','pred_pheno', 'pred_sex_age', 'pred_cov'],axis=0)
# q_combined

# # %%
# q_combined.to_csv(save_path+'UKB_cox_quantile_predicted_diff.csv')

# # %% [markdown]
# # Cancer death

# # %%
# # using diff
# pred_age = hr_regression(['predicted'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(pred_age)
# pheno_age = hr_regression(['label'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(pheno_age)
# fi_age = hr_regression(['frailty index'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(fi_age)
# pred_pheno = hr_regression(['predicted','phenoage'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(pred_pheno)
# pred_sex_age = hr_regression(['predicted','baseline_age','Sex'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(pred_sex_age)
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
# pred_cov = hr_regression(cov_col,ehr_age_map_all,category_to_dummy=category_to_dummy ,death_label='Cancer_Flag')
# # print(pred_cov)

# # %%
# combined = pd.concat([pred_age, pheno_age, fi_age ,pred_pheno, pred_sex_age, pred_cov],
#     keys=['pred_age','pheno_age','frailty index' ,'pred_pheno', 'pred_sex_age', 'pred_cov'],axis=0)
# combined.to_csv(save_path+'UKB_cox_regession_predicted_diff_cancer.csv')

# # %%
# # using qunatile
# pred_age_q = hr_quantile('predicted',['predicted'], ehr_age_map_all, death_label='Cancer_Flag')
# # print(pred_age_q)
# phenoage_q = hr_quantile('label',['label'], ehr_age_map_all, death_label='Cancer_Flag')
# # print(phenoage_q)
# fi_age_q = hr_quantile('frailty index', ['frailty index'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(fi_age_q)
# pred_pheno_q = hr_quantile('predicted',['predicted','phenoage'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(pred_age_q)
# pred_sex_age_q = hr_quantile('predicted',['predicted','baseline_age','Sex'],ehr_age_map_all, death_label='Cancer_Flag')
# # print(pred_sex_age_q)
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
# pred_cov_q = hr_quantile('predicted', cov_col,ehr_age_map_all,category_to_dummy=category_to_dummy , death_label='Cancer_Flag')
# # print(pred_cov_q)

# # %%
# q_combined = pd.concat([pred_age_q,phenoage_q, fi_age_q,pred_pheno_q, pred_sex_age_q, pred_cov_q],
#     keys=['pred_age_diff', 'phenoage', 'frailty index','pred_pheno', 'pred_sex_age', 'pred_cov'],axis=0)
# q_combined.to_csv(save_path+'UKB_cox_quantile_predicted_diff_cancer.csv')

# # %%
# # using diff
# pred_age = hr_regression(['predicted'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(pred_age)
# pheno_age = hr_regression(['label'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(pheno_age)
# fi_age = hr_regression(['frailty index'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(fi_age)
# pred_pheno = hr_regression(['predicted','phenoage'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(pred_pheno)
# pred_sex_age = hr_regression(['predicted','baseline_age','Sex'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(pred_sex_age)
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
# pred_cov = hr_regression(cov_col,ehr_age_map_all,category_to_dummy=category_to_dummy , death_label='Cardio_Flag')
# # print(pred_cov)

# # %%
# combined = pd.concat([pred_age, pheno_age,fi_age ,pred_pheno, pred_sex_age, pred_cov],
#     keys=['pred_age','pheno_age','frailty index' , 'pred_pheno', 'pred_sex_age', 'pred_cov'],axis=0)
# combined.to_csv(save_path+'UKB_cox_regession_predicted_diff_CVD.csv')

# # %%
# # using qunatile
# pred_age_q = hr_quantile('predicted',['predicted'], ehr_age_map_all, death_label='Cardio_Flag')
# # print(pred_age_q)
# phenoage_q = hr_quantile('label',['label'], ehr_age_map_all, death_label='Cardio_Flag')
# # print(phenoage_q)
# fi_age_q = hr_quantile('frailty index', ['frailty index'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(fi_age_q)
# pred_pheno_q = hr_quantile('predicted',['predicted','phenoage'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(pred_age_q)
# pred_sex_age_q = hr_quantile('predicted',['predicted','baseline_age','Sex'],ehr_age_map_all, death_label='Cardio_Flag')
# # print(pred_sex_age_q)
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking','IMD','ethnicity']
# pred_cov_q = hr_quantile('predicted', cov_col,ehr_age_map_all, category_to_dummy=category_to_dummy ,death_label='Cardio_Flag')
# # print(pred_cov_q)

# # %%
# q_combined = pd.concat([pred_age_q,phenoage_q, fi_age_q ,pred_pheno_q, pred_sex_age_q, pred_cov_q],
#     keys=['pred_age_diff', 'phenoage', 'frailty index', 'pred_pheno', 'pred_sex_age', 'pred_cov'],axis=0)
# q_combined.to_csv(save_path+'UKB_cox_quantile_predicted_diff_CVD.csv')

# %% [markdown]
#  # CPRD!!!!

# %%
path = ''
cprd_age = pd.read_csv(path)
cprd_death = pd.read_csv('data/cprd_death_withreason_updated.csv')[['patid', 'event', 'time2death', 'Cancer_Flag', 'Cardio_Flag']]
cprd_age_all = cprd_death.merge(cprd_age, on='patid', how='inner')
cov = cov_cprd.rename(columns={'eid':'patid'})
cprd_age_all = cprd_age_all.merge(cov, on='patid', how='inner')
print('all data:', cprd_age_all.shape)


# %%
# # using diff
pred_age = hr_regression(['predicted'],cprd_age_all)
pred_sex_age = hr_regression(['predicted','baseline_age','Sex'],cprd_age_all)
cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking','IMD','ethnicity']
pred_cov = hr_regression(cov_col,cprd_age_all, category_to_dummy=category_to_dummy ,penalizer=0.05)
combined = pd.concat([pred_age,  pred_sex_age, pred_cov],
    keys=['pred_age', 'pred_sex_age', 'pred_cov'],axis=0)
combined.to_csv(save_path+'CPRD_cox_regession_predicted_diff.csv')


# %%
# pred_age_q = hr_quantile('predicted',['predicted'], cprd_age_all)
# pred_sex_age_q = hr_quantile('predicted',['predicted','baseline_age','Sex'],cprd_age_all)
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
# pred_cov_q = hr_quantile('predicted', cov_col,cprd_age_all,category_to_dummy=category_to_dummy , penalizer=0.05)
# q_combined = pd.concat([pred_age_q, pred_sex_age_q, pred_cov_q],
#     keys=['pred_age_diff',  'pred_sex_age', 'pred_cov'],axis=0)

# # %%
# q_combined.to_csv(save_path+'CPRD_cox_quantile_predicted_diff.csv')

# %% [markdown]
# Cancer

# %%
# # using diff
pred_age = hr_regression(['predicted'],cprd_age_all, death_label='Cancer_Flag')
# print(pred_pheno)
pred_sex_age = hr_regression(['predicted','baseline_age','Sex'],cprd_age_all, death_label='Cancer_Flag')
# print(pred_sex_age)
cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
pred_cov = hr_regression(cov_col,cprd_age_all, category_to_dummy=category_to_dummy, penalizer=0.05, death_label='Cancer_Flag')
# print(pred_cov)
combined = pd.concat([pred_age,  pred_sex_age, pred_cov],
    keys=['pred_age', 'pred_sex_age', 'pred_cov'],axis=0)
combined.to_csv(save_path+'CPRD_cox_regession_predicted_diff_cancer.csv')


# # %%
# pred_age_q = hr_quantile('predicted',['predicted'], cprd_age_all, death_label='Cancer_Flag')
# pred_sex_age_q = hr_quantile('predicted',['predicted','baseline_age','Sex'],cprd_age_all, death_label='Cancer_Flag')
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
# pred_cov_q = hr_quantile('predicted', cov_col,cprd_age_all,category_to_dummy=category_to_dummy , penalizer=0.05, death_label='Cancer_Flag')
# q_combined = pd.concat([pred_age_q, pred_sex_age_q, pred_cov_q],
#     keys=['pred_age_diff',  'pred_sex_age', 'pred_cov'],axis=0)
# q_combined.to_csv(save_path+'CPRD_cox_quantile_predicted_diff_cancer.csv')


# %% [markdown]
# CVD

# %%
# # using diff
pred_age = hr_regression(['predicted'],cprd_age_all, death_label='Cardio_Flag')
# print(pred_pheno)
pred_sex_age = hr_regression(['predicted','baseline_age','Sex'],cprd_age_all, death_label='Cardio_Flag')
# print(pred_sex_age)
cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking','IMD','ethnicity']
pred_cov = hr_regression(cov_col,cprd_age_all, category_to_dummy=category_to_dummy ,penalizer=0.05, death_label='Cardio_Flag')
# print(pred_cov)
combined = pd.concat([pred_age,  pred_sex_age, pred_cov],
    keys=['pred_age', 'pred_sex_age', 'pred_cov'],axis=0)
combined.to_csv(save_path+'CPRD_cox_regession_predicted_diff_CVD.csv')

# %%
# pred_age_q = hr_quantile('predicted',['predicted'], cprd_age_all, death_label='Cardio_Flag')
# pred_sex_age_q = hr_quantile('predicted',['predicted','baseline_age','Sex'],cprd_age_all, death_label='Cardio_Flag')
# cov_col =['predicted','baseline_age','Sex', 'BMI', 'Systolic_Blood_Pressure', 'HDL_Cholesterol', 'LDL_Cholesterol', 'Smoking', 'IMD','ethnicity']
# pred_cov_q = hr_quantile('predicted', cov_col,cprd_age_all,category_to_dummy=category_to_dummy , penalizer=0.05, death_label='Cardio_Flag')
# q_combined = pd.concat([pred_age_q, pred_sex_age_q, pred_cov_q],
#     keys=['pred_age_diff',  'pred_sex_age', 'pred_cov'],axis=0)
# q_combined.to_csv(save_path+'CPRD_cox_quantile_predicted_diff_CVD.csv')



