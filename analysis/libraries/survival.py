from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.statistics import pairwise_logrank_test
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

def plot_km_two_groups(df, event_ind_field, event_data_field, group_col, max_months, add_counts=False, ci_show=False, title='', label_neg_class='Low risk', label_pos_class='High risk', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    time = np.linspace(0, max_months)
    high_colour = 'tab:blue'
    low_colour = 'tab:orange'
    kmf_l = KaplanMeierFitter(label=label_neg_class)
    kmf_l.fit(df[df[group_col] == 0][event_data_field], df[df[group_col] == 0][event_ind_field], timeline=time)
    kmf_h = KaplanMeierFitter(label=label_pos_class)
    kmf_h.fit(df[df[group_col] == 1][event_data_field], df[df[group_col] == 1][event_ind_field], timeline=time)
    kmf_l.plot_survival_function(show_censors=True, ci_show=ci_show, ax=ax, lw=2, color=low_colour)
    kmf_h.plot_survival_function(show_censors=True, ci_show=ci_show, ax=ax, lw=2, color=high_colour)

    if add_counts:
        # add_at_risk_counts(kmf_l, kmf_h, rows_to_show=['At risk'], ax=ax)
        add_at_risk_counts(kmf_l, kmf_h, ax=ax)

    result = logrank_test(df[df[group_col] == 1][event_data_field].values, df[df[group_col] == 0][event_data_field].values, df[df[group_col] == 1][event_ind_field].values, df[df[group_col] == 0][event_ind_field].values)
    p_val = np.round(result.p_value, 3)
    if p_val < 0.001:
        p_val = 'p < 0.001'
    else:
        p_val = f'p = {p_val}'

    ax.set_title(f'{title}')
    ax.text(x=3, y=0.05, s=f'Log rank test \n {p_val}')
    ax.set_ylim([0.1,1.10])
    ax.set_ylabel('Survival Probability\n (Overall Survival)')
    ax.set_xlabel('Timeline (Months)')
    ax.set_xticks(ticks=ax.get_xticks())
    ax.set_yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend()
    if max_months is None:
        max_months = df[event_data_field].values.max() + 6
    ax.set_xlim([0.0, max_months])
    plt.tight_layout()