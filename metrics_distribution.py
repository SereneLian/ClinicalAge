"""Gap distributions/summaries and AGE prediction metrics.
Requires pandas, numpy, matplotlib and scikit-learn.
metrics_report compares predicted_age with gt_label
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def plot_gap_distribution(df, gap_col='predicted', bins=20, ax=None):
    """Plot the supplied gap column; return (figure, axes)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[gap_col].dropna(), bins=bins)
    ax.set(title=f'Distribution of {gap_col}', xlabel='Age gap (years)', ylabel='Count')
    return ax.figure, ax


def summary_median_iqr_by(df, value_col='predicted', group_col=None):
    """Overall or grouped summary, using gap."""
    cols = [value_col] + ([group_col] if group_col is not None else [])
    data = df[cols].dropna()
    if group_col is not None:
        result = data.groupby(group_col, observed=True)[value_col].agg(
            N='count', Median='median',
            Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75)
        ).reset_index()
    else:
        x = data[value_col]
        result = pd.DataFrame([{
            'N': len(x), 'Median': x.median(),
            'Q1': x.quantile(0.25), 'Q3': x.quantile(0.75)
        }])
    result['IQR'] = result['Q3'] - result['Q1']
    return result


def bland_altman_stats(x, y):
    """Return mean difference and 95% limits of agreement."""
    diff = x - y
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    loa_low = md - 1.96 * sd
    loa_high = md + 1.96 * sd
    return md, loa_low, loa_high, diff

def bootstrap_ci(data, metric_func, n_bootstraps=1000, alpha=0.05, random_state=None):
    """Percentile CI from paired row resampling."""
    if len(data) == 0 or n_bootstraps < 1 or not 0 < alpha < 1:
        raise ValueError('Require nonempty data, n_bootstraps >= 1 and 0 < alpha < 1.')
    rng = np.random.RandomState(random_state)
    bootstrapped_metrics = []
    for _ in range(n_bootstraps):
        sample = data.sample(len(data), replace=True, random_state=rng)
        bootstrapped_metrics.append(metric_func(sample))
    return tuple(np.percentile(bootstrapped_metrics, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def metrics_report(ehr_age_map, dataset='', gt_label='phenoage',
                   n_bootstraps=1000, alpha=0.05, random_state=None):
    """Compare AGE values: predicted_age versus gt_label.
    R²/Pearson need a nonconstant reference; MAPE needs nonzero reference ages.
    Undefined bootstrap metrics propagate NaN to their CIs, as in the original.
    """
    data = ehr_age_map[[gt_label, 'predicted', 'baseline_age']].copy()
    data['predicted_age'] = data['predicted'] +data['baseline_age']
    if len(data) < 2:
        raise ValueError('At least two complete age pairs are required.')

    def r2_metric(df):
        return r2_score(df[gt_label], df['predicted_age'])

    def pearson_metric(df):
        return df['predicted_age'].corr(df[gt_label], method='pearson')

    def mae_metric(df):
        return mean_absolute_error(df[gt_label], df['predicted_age'])

    def mse_metric(df):
        return mean_squared_error(df[gt_label], df['predicted_age'])

    def mape_metric(df):
        return np.mean(np.abs((df[gt_label] - df['predicted_age']) / df[gt_label])) * 100

    def pred_diff_metric(df):
        return (df[gt_label] - df['predicted_age']).mean()

    metrics = {
        'R²': r2_metric,
        'Pearson Correlation': pearson_metric,
        'Mean Absolute Error': mae_metric,
        'Mean Squared Error': mse_metric,
        'Mean Absolute Percentage Error': mape_metric,
        'Mean Predicted Age Difference': pred_diff_metric,
    }
    rows = []
    for name, metric_func in metrics.items():
        value = metric_func(data)
        lower, upper = bootstrap_ci(data, metric_func, n_bootstraps, alpha, random_state)
        rows.append({'Dataset': dataset, 'N': len(data), 'Metric': name, 'Value': value,
                     'CI Lower Bound': lower, 'CI Upper Bound': upper})
        print(f'{name}: {value:.4f} ({100 * (1 - alpha):g}% CI: {lower:.4f} - {upper:.4f})')
    return pd.DataFrame(rows)
