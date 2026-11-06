#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:41:18 2026

@author: nat
"""
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import os.path as op
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA 
from sklearn.dummy import DummyRegressor
from nilearn import image
from collections import defaultdict
#%matplotlib qt
#%% FUNCTIONS

def extract_region(feature_name):
    for suf in suffixes:
        if feature_name.endswith(suf):
            return feature_name[:-(len(suf) + 1)]  # remove '_' + suffix
 
def extract_suffix(feature_name):
    for suf in suffixes:
        if feature_name.endswith(suf):
            return suf
#%% ####################### FEATURE MODEL #############################

# Load excel file of features
Data = pd.read_excel('/d/mjt/9/projects/OPM-Analysis/OPM_BE_vs_IS/analysis/powerspec/BEvIS_brainvalues.xlsx',sheet_name=None)
Data.pop('GlobalConn',None)
#Data.pop('Alpha_Peak_Amp',None)
# Merge data into pandas dataframe
common_cols = ['Subject','Age','Sex', 'HM', 'Trials','Task']
df_merged = Data['Slope'].copy()

for name,df in Data.items():
    if name == 'Slope':
        continue   
    df_to_add = df.drop(columns = common_cols)
    df_merged = pd.concat([df_merged, df_to_add],axis=1)
      
# Separate into baby einsteins and inscapes data
df_babyeinstein = df_merged[df_merged['Task'] == 'babyeinstein'].copy()
df_inscapes = df_merged[df_merged['Task'] == 'inscapes'].copy()
# Get only feature columns
feature_cols = [c for c in df_babyeinstein if c not in ['Subject','Age','Sex','HM','Trials','Task']]


#%% Train-Test on different tasks

# X are the features, y are the ages
# Baby einsteins
X_be = df_babyeinstein[feature_cols].values
y_be = df_babyeinstein['Age'].values

# Inscapes
X_in = df_inscapes[feature_cols].values
y_in = df_inscapes['Age'].values


# Make the model pipeline, imputing missing values, scaling and using ridge regression
model = make_pipeline(
    SimpleImputer(),
    StandardScaler(),
    Ridge(alpha=100)
)

# Train on BE, Test on BE with 5-fold cross validation
pred_be = cross_val_predict(model, X_be, y_be, cv=5)
mae_be = mean_absolute_error(y_be, pred_be)
r2_be = r2_score(y_be, pred_be)
print(mae_be, r2_be)

# Train on Inscapes, Test on Inscapes with 5-fold cross validation
pred_in = cross_val_predict(model, X_in, y_in, cv=5)
mae_in = mean_absolute_error(y_in, pred_in)
r2_in = r2_score(y_in, pred_in)
print(mae_in, r2_in)


# Train on BE, Test on Inscapes
model.fit(X_be, y_be)
pred_be2in = model.predict(X_in)
mae_be2in = mean_absolute_error(y_in, pred_be2in)
r2_be2in = r2_score(y_in, pred_be2in)
print(mae_be2in, r2_be2in)

# Train on Inscapes, Test on BE
model.fit(X_in, y_in)
pred_in2be = model.predict(X_be)
mae_in2be = mean_absolute_error(y_be, pred_in2be)
r2_in2be = r2_score(y_be, pred_in2be)
print(mae_in2be, r2_in2be)


#%% Dummy model for comparison

dummy = DummyRegressor(strategy='mean')
pred_dummy = cross_val_predict(dummy, X_be, y_be, cv=5)
mae_dummy = mean_absolute_error(y_be, pred_dummy)
r2_dummy = r2_score(y_be, pred_dummy)
print(mae_dummy, r2_dummy)

#%% Get most predictive features
suffixes = [
    'slope',
    'offset',
    'theta',
    'alpha',
    'beta',
    'lowgamma',
    'highgamma',
    'alpha_peak_freq',
    'alpha_peak_amp']#,
    #'GlobalConn']

       
regions = [extract_region(name) for name in feature_cols]    
suffix_all = [extract_suffix(name) for name in feature_cols]

############ INSCAPES ##########################################
model.fit(X_in, y_in)
ridge = model.named_steps['ridge']
coefs = ridge.coef_
coefs_features_in = coefs
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coef': coefs,
    'abs_coef': np.abs(coefs)
}).sort_values('abs_coef', ascending=False)

print(f"Inscapes training top 20 features: {coef_df.head(20)}")
region_dict = defaultdict(list)

for reg, coef in zip(regions, coefs):
    region_dict[reg].append(coef)

region_importance = {
    reg: np.mean(np.abs(vals)) for reg, vals in region_dict.items()
}

region_vals_in = list(region_importance.items())
region_vals_in = [v for k,v in region_vals_in[1:]]

region_vals_in_min = (min(region_vals_in) + max(region_vals_in))/2

suffix_dict = defaultdict(list)

for suf, coef in zip(suffix_all, coefs):
    suffix_dict[suf].append(coef)

suffix_importance = {
    suf: float(np.mean(np.abs(vals))) for suf, vals in suffix_dict.items()
}

feature_vals_in = list(suffix_importance.items())
feature_labels_in = [k for k,v in feature_vals_in]
feature_vals_in = [v for k,v in feature_vals_in]


plt.figure()
plt.bar(feature_labels_in, feature_vals_in)
plt.xticks(rotation=45)
plt.ylabel('Mean absolute coefficient')
plt.title('Feature Importance Inscapes')
plt.tight_layout()
plt.show()




################ BABY EINSTEINS ##########################
model.fit(X_be, y_be)

ridge = model.named_steps['ridge']
coefs = ridge.coef_
coefs_features_be = coefs
coef_df = pd.DataFrame({
    'feature': feature_cols,
    'coef': coefs,
    'abs_coef': np.abs(coefs)
}).sort_values('abs_coef', ascending=False)
print(f"Baby Einsteins training top 20 features: {coef_df.head(20)}")
region_dict = defaultdict(list)

for reg, coef in zip(regions, coefs):
    region_dict[reg].append(coef)

region_importance = {
    reg: np.mean(np.abs(vals)) for reg, vals in region_dict.items()
}

region_vals_be = list(region_importance.items())
region_vals_be = [v for k,v in region_vals_be[1:]]

region_vals_be_min = (min(region_vals_be) + max(region_vals_be))/2

suffix_dict = defaultdict(list)

for suf, coef in zip(suffix_all, coefs):
    suffix_dict[suf].append(coef)

suffix_importance = {
    suf: float(np.mean(np.abs(vals))) for suf, vals in suffix_dict.items()
}

feature_vals_be = list(suffix_importance.items())
feature_labels_be = [k for k,v in feature_vals_be]
feature_vals_be = [v for k,v in feature_vals_be]


plt.figure()
plt.bar(feature_labels_be, feature_vals_be)
plt.xticks(rotation=45)
plt.ylabel('Mean absolute coefficient')
plt.title('Feature importance Baby Einsteins')
plt.tight_layout()
plt.show()


#%% Make matrix plots for MAE and R2


fig, axes = plt.subplots(1,2, figsize=(5,2.5))
fig.subplots_adjust(wspace=0.4)
plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.5
})
# RIGHT PANEL: MAE Bar Plot


mae = np.array([
    [mae_be,     mae_be2in],   # train Baby Einstein
    [mae_in2be,  mae_in]       # train Inscapes
])

r2 = np.array([
    [r2_be,      r2_be2in],    # train Baby Einstein
    [r2_in2be,   r2_in]        # train Inscapes
])

row_labels = ['Baby Einstein', 'Inscapes']
col_labels = ['Baby Einstein', 'Inscapes']
# Create two inset axes
ax_mae = axes [0]
ax_r2  = axes[1]
# ---------------------------------------------------------
# MAE matrix
# ---------------------------------------------------------

# Reverse the colormap so darker = lower/better MAE
im1 = ax_mae.imshow(
    mae,
    cmap='Purples_r',
    vmin=0.50,
    vmax=1
)

ax_mae.set_xticks([0, 1])
ax_mae.set_xticklabels(col_labels, fontsize=9)
ax_mae.set_yticks([0, 1])
ax_mae.set_yticklabels(row_labels, fontsize=9,rotation=90, ha='center', va='center')
ax_mae.tick_params(axis='y',pad=6)
ax_mae.set_xlabel('Test', fontsize=12)
ax_mae.set_ylabel('Train', fontsize=12)
ax_mae.set_title('MAE (years)', fontsize=12)

# Add values
for i in range(2):
    for j in range(2):
        ax_mae.text(
            j, i, f'{mae[i, j]:.2f}',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold'
        )
# Here normal Greens = higher/better
im2 = ax_r2.imshow(
    r2,
    cmap='Purples',
    vmin=0.2,
    vmax=0.75
)

ax_r2.set_xticks([0, 1])
ax_r2.set_xticklabels(col_labels, fontsize=9)
ax_r2.set_yticks([0, 1])
ax_r2.set_yticklabels(row_labels, fontsize=9,rotation=90, ha='center', va='center')
ax_r2.tick_params(axis='y',pad=6)
ax_r2.set_ylabel('Train', fontsize=12)
ax_r2.set_xlabel('Test', fontsize=12)
ax_r2.set_title(r'$R^2$', fontsize=12)

# Add values
for i in range(2):
    for j in range(2):
        ax_r2.text(
            j, i, f'{r2[i, j]:.2f}',
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold'
        )

# Remove spines
for a in [ax_mae, ax_r2]:
    for spine in a.spines.values():
        spine.set_visible(False)


plt.savefig('/d/mjt/9/projects/OPM-Analysis/OPM_BE_vs_IS/analysis/brainage/brain_age_results_maer2.png', dpi=300, transparent=True, bbox_inches='tight')

plt.show()



#%%########## CORRELATE FEATURE IMPORTANCE ##############

result = spearmanr(coefs_features_in,coefs_features_be)
print(f"Statistic = {result.statistic}, p-value = {result.pvalue}")




#%% Make predicted vs real age and feature correlation plots

fig, axes = plt.subplots(1, 2, figsize=(7,4))
fig.subplots_adjust(wspace=0.4)
plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.5
})

color_be = 'tab:blue'
color_in = 'tab:orange'

# LEFT PANEL: Predicted vs Real Age

ax = axes[0]

# Scatter points
ax.scatter(y_be, pred_be, alpha=0.6, color=color_be,
           label=f'Baby Einstein')
ax.scatter(y_in, pred_in, alpha=0.6, color=color_in,
           label=f'Inscapes')

# BE
m_be, b_be = np.polyfit(y_be, pred_be, 1)
x_be_line = np.linspace(y_be.min(), y_be.max(), 100)
ax.plot(x_be_line, m_be * x_be_line + b_be, color=color_be, linewidth=2)

# # IS
m_in, b_in = np.polyfit(y_in, pred_in, 1)
x_in_line = np.linspace(y_in.min(), y_in.max(), 100)
ax.plot(x_in_line, m_in * x_in_line + b_in, color=color_in, linewidth=2)


ax.set_xlabel('Real age (years)')
ax.set_ylabel('Predicted age (years)')
ax.set_title('Predicted vs Real Age')
#ax.legend()
ax.set_xlim(0.5,6.5)
ax.set_ylim(0.5,6.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', adjustable='box')

# feature correlation scatter plot

ax = axes[1]
plt.rcParams.update({
    'font.size': 13,
    'axes.linewidth': 1.5
})
# 1. Plot the feature scatter points
# Using an alpha < 1 helps when hundreds of points overlap (creates a 'density' look)
ax.scatter(coefs_features_be, coefs_features_in, color='#2c3e50', alpha=0.9, edgecolors='none', s=40, label='Brain Features')

# 2. CALCULATE & PLOT THE LINE OF BEST FIT (Linear)
# Calculate the slope (m) and intercept (b)
m, b = np.polyfit(coefs_features_be, coefs_features_in, 1)

# Create an array of X values spanning the range of your data
x_line = np.linspace(min(coefs_features_be), max(coefs_features_be), 100)
y_line = m * x_line + b

ax.plot(x_line, y_line, color='k', linewidth=2.5, label='Trend Line')

ax.set_ylabel('Inscapes Feature ($\\beta$)')
ax.set_xlabel('Baby Einstein Feature ($\\beta$)')
ax.set_title('Feature Correlation')
# 5. Clean up the Box/Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save out safely as a transparent high-res asset
plt.savefig('/d/mjt/9/projects/OPM-Analysis/OPM_BE_vs_IS/analysis/brainage/pred_real_andfeaturecorr_graphabs.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()