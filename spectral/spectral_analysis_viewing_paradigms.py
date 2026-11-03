# -*- coding: utf-8 -*-
"""
Spectral parameterization of resting-state OPM-MEG power spectra.

Fits specparam (formerly FOOOF) models to source-level power spectra from two
resting-state conditions: Baby Einstein and Inscapes.

Inputs
------
- Power spectra exported from MATLAB after epoching, filtering, artifact
  rejection, and source modelling
  Shape: n_regions x n_frequencies x n_participants
- Participant demographics spreadsheet

Author: Julian Bandhan
Created: 2025-03-13
Updated: 2026-11-02
"""
# %% 0 - Funcs
import importlib
from functions.common_imports import np, plt, stats, sns, pd, loadmat, image, Line2D
from functions.brainplots import make_atlas_nifti, surface_brain_plotv2, make_4d_atlas_nifti
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel
from matplotlib.ticker import MaxNLocator
import os

# ── Colour palette ──────────────────────────────────────────────────────────
col_redpurple = "#CC3399"
col_purple    = "#400080"
col_green     = "#00802b"
col_orange    = "#ff6c00"
col_red       = "#ff0000"

FS_DIR = "path/to/freesurfer/subjects"



def shade_bands(ax, band_param='low'):
    """Apply standard frequency-band shading and dotted dividers to ax."""
    _shade(ax, -np.inf, 6,  col_redpurple)
    _shade(ax, 6,  13,      col_purple)
    _shade(ax, 13, 30,      col_green)
    _shade(ax, 30, 55,      col_orange)
    _shade(ax, 65, 80,      col_red)
    _shade(ax, 55, 65, color="#ffffff", hatch='////', zorder=0,
           hatch_ec="#000000", hatch_lw=0)

    vlines = {
        'low'  : [6, 13],
        'all'  : [6, 13, 30, 55, 65],
        'gamma': [55, 65],
    }
    for vx in vlines.get(band_param, []):
        ax.axvline(x=vx, linestyle='dotted', linewidth=2, color='k')

def _shade(ax, a, b, color, alpha=0.1, hatch=None, zorder=0,
           hatch_ec="#000000", hatch_lw=0):
    """Applies Shade to range in shade_bands"""
    from matplotlib.colors import to_rgb
    xmin, xmax = ax.get_xlim()
    a, b = max(a, xmin), min(b, xmax)
    if b <= a:
        return
    if hatch is None:
        ax.axvspan(a, b, facecolor=color, alpha=alpha,
                   edgecolor='none', zorder=zorder)
    else:
        rgba  = (*to_rgb(color), 0.0)
        patch = ax.axvspan(a, b, facecolor=rgba, edgecolor=hatch_ec,
                           alpha=0.3, hatch=hatch, zorder=zorder)
        patch.set_linewidth(hatch_lw)
        
from matplotlib.ticker import ScalarFormatter

class FixedSciFormatter(ScalarFormatter):
    """MATLAB-style axis: x10^n offset on top, fixed-decimal mantissa on ticks."""
    def __init__(self, decimals=1, sci_limits=(-2, 3)):
        super().__init__(useMathText=True)      # renders x10^{-3}, not 1e-3
        self.decimals = decimals
        self.set_powerlimits(sci_limits)        # sci notation outside 10^-2 .. 10^3

    def _set_format(self):
        self.format = r'$\mathdefault{%%1.%df}$' % self.decimals
        
# ---- Plot function ----
def plot_paired_comparison(be_vals, is_vals, ylabel, title, jitter=0.08,
                           sci_decimals=1, sci_limits=(-2, 3)):
    """Plot comparison scatter plots with box plots"""
    fig, ax = plt.subplots(figsize=(5, 8))

    # Jittered x positions
    x_be = np.zeros(len(be_vals)) + np.random.uniform(-jitter, jitter, size=len(be_vals))
    x_is = np.ones(len(is_vals))  + np.random.uniform(-jitter, jitter, size=len(is_vals))

    # Draw paired lines first (behind scatter)
    for xb, xi, bv, iv in zip(x_be, x_is, be_vals, is_vals):
        ax.plot([xb, xi], [bv, iv], color='0.7', linewidth=0.8, zorder=1)

    # Scatter
    ax.scatter(x_be, be_vals, alpha=0.7, s=100, color='tab:blue',   zorder=2)
    ax.scatter(x_is, is_vals, alpha=0.7, s=100, color='tab:orange', zorder=2)

    # Box plots 
    bp = ax.boxplot(
        [be_vals, is_vals],
        positions=[0, 1],
        widths=0.15,
        patch_artist=True,
        showfliers = False,
        manage_ticks=False,
        zorder=3,
        boxprops=dict(color='0.3', linewidth=2),
        medianprops=dict(color='0.3', linewidth=2.5),
        whiskerprops=dict(color='0.3', linewidth=2),
        capprops=dict(color='0.3', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='none',
                        markeredgecolor='black', markersize=8, linewidth=1.5),
    )
    
    import matplotlib.colors as mcolors

    colors = ['tab:blue', 'tab:orange']
    for patch, color in zip(bp['boxes'], colors):
        rgba = mcolors.to_rgba(color, alpha=0.3)
        patch.set_facecolor(rgba)
        patch.set_edgecolor('0.3')
        patch.set_linewidth(2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BE', 'IS'])
    ax.set_ylabel(ylabel, fontsize=65)
    ax.set_xlabel(title,  fontsize=35)
    ax.tick_params(axis='both', labelsize=50)

    # MATLAB-style y axis: x10^n on top, fixed decimals on the ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FixedSciFormatter(sci_decimals, sci_limits))
    ax.yaxis.get_offset_text().set_fontsize(40)   # match tick label size

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    return fig
        
# ---- Other helpers ----
def paired_t_for_region(df, region, band_col):
    """Paired t-test for one region and band using pivot to enforce matching."""
    sub  = df[df['Region'] == region]
    wide = sub.pivot(index='Subject', columns='Task', values=band_col)
    if ('babyeinstein' not in wide.columns) or ('inscapes' not in wide.columns):
        return np.nan, np.nan
    paired = wide[['babyeinstein', 'inscapes']].dropna()
    if len(paired) < 2:
        return np.nan, np.nan
    a = paired['babyeinstein'].to_numpy(dtype=float)
    b = paired['inscapes'].to_numpy(dtype=float)
    t, p = ttest_rel(a, b, nan_policy='omit')
    return float(t), float(p)

def fdr_nan_safe(pvals, alpha=0.05):
    """BH-FDR correction ignoring NaNs. Returns (corrected_pvals, reject_mask)."""
    p      = np.asarray(pvals, dtype=float)
    finite = np.isfinite(p)
    p_corr = np.full_like(p, np.nan)
    reject = np.zeros(len(p), dtype=bool)
    if finite.sum() > 0:
        rej, p_c, _, _ = multipletests(p[finite], alpha=alpha, method='fdr_bh')
        p_corr[finite] = p_c
        reject[finite] = rej
    return p_corr, reject

def top_table(pvals, tstats, regions, k=5):
    return (pd.DataFrame({'Region': regions, 't': tstats, 'p': pvals})
            .sort_values('p', na_position='last')
            .head(k))

def paired_t_wholebrain(wb_df, band_col):
    """Paired t-test on whole-brain mean, matched on Subject."""
    wide   = wb_df.pivot(index='Subject', columns='Task', values=band_col)
    if ('babyeinstein' not in wide.columns) or ('inscapes' not in wide.columns):
        return np.full(len(wb_df), np.nan), np.full(len(wb_df), np.nan), np.nan, np.nan
    paired = wide[['babyeinstein', 'inscapes']].dropna()
    if len(paired) < 2:
        return np.full(len(wb_df), np.nan), np.full(len(wb_df), np.nan), np.nan, np.nan
    a = paired['babyeinstein'].to_numpy(dtype=float)
    b = paired['inscapes'].to_numpy(dtype=float)
    t, p = ttest_rel(a, b, nan_policy='omit')
    return a, b, float(t), float(p)

def build_marker_df(marker, wb_vals, reg_vals, valid_range=None):
    """
    marker      : str, used as column suffix
    wb_vals     : (num_sub,) whole-brain value per subject row
    reg_vals    : (num_sub, num_reg) per-region values
    valid_range : optional (lo, hi); values outside [lo, hi] -> NaN
    """
    if valid_range is not None:
        lo, hi = valid_range
        wb_vals  = np.asarray(wb_vals,  dtype=float).copy()
        reg_vals = np.asarray(reg_vals, dtype=float).copy()
        wb_vals[(wb_vals  < lo) | (wb_vals  > hi)] = np.nan
        reg_vals[(reg_vals < lo) | (reg_vals > hi)] = np.nan

    base = pd.DataFrame({
        'Subject':              subjects,
        'Age':                  ages,
        'Sex':                  prime_sex,
        'HM':                   hm,
        'Trials':               trials,
        'Task':                 tasks,
        f'WholeBrain_{marker}': wb_vals,
    })
    reg_df = pd.DataFrame(
        reg_vals,
        columns=[f'{r}_{marker}' for r in reg_name]
    )
    return pd.concat([base, reg_df], axis=1)


# %% 1 - Load in the Data
"""
Load PSD data, subject data, atlas files, and region labels.
"""

# Input and output paths
atlasimg = "path/to/atlas/atlas_4D.nii.gz"  # 4D atlas
atlasimg3d = "path/to/atlas/atlas_3D.nii"  # 3D atlas
psd_file = "path/to/data/psd_data.mat"
demographics_file = "path/to/demographics/demographics.xlsx"
labels_file = "path/to/atlas/atlas_labels.txt"
labels_full_file = "path/to/atlas/atlas_labels_full.txt"
base_dir = "path/to/output"

plt.rcParams['font.family'] = 'Arial'
os.makedirs(base_dir, exist_ok=True)

# Load PSD data and demographics
prime_data = loadmat(psd_file)['data']
subject_data = pd.read_excel(demographics_file)

# MATLAB saves a 1x1 struct as a nested array, so extract the struct
if isinstance(prime_data, np.ndarray):
    prime_data = prime_data[0, 0]

# Row indices of scans to exclude (leave empty to keep all scans)
# Assumes the PSD file and demographics table list scans in the same order
drop_idx = []
subject_data = subject_data.drop(index=drop_idx).reset_index(drop=True)

# Subject IDs from the PSD file (format depends on how MATLAB saved them)
subjects = prime_data['subject']
subjects = np.delete(subjects, drop_idx, axis=0)

if isinstance(subjects, np.ndarray):
    if subjects.dtype.type is np.object_:  # MATLAB cell array
        subjects = [str(s[0]) for s in subjects]

# Remove leftover brackets and quotes from the IDs
subjects = [s.strip("[]'") for s in subjects]

# Stop if the PSD file and demographics table have different scan counts
assert len(subjects) == len(subject_data), "PSD and demographics scan counts differ"

# Build a unique ID for each scan: subject-sess-XX-run-XX
subject_ids = subject_data['subject'].str.strip("[]'")
subject_visitId = [
    f"{subj}-sess-{int(sess):02}-run-{int(run):02}"
    for subj, sess, run in zip(subject_ids, subject_data['session'], subject_data['run'])
]

# Sex
prime_sex = subject_data['sex'].values

# Task labels, with spelling standardized
tasks = subject_data['task']
tasks = ['babyeinstein' if t == 'babyeinsteins' else t for t in tasks]

# Frequency values shared by all scans
prime_freq = prime_data['freq'][:, 0]

# PSD data, with scans on axis 0
prime_psd = prime_data['psd']
prime_psd = np.delete(prime_psd, drop_idx, axis=0)

# Age in years
ages = prime_data['age'][:, 0]
ages = np.delete(ages, drop_idx, axis=0)

# Age bins: ages up to 1 go in bin 1, older ages round down to whole years
age_bin = np.where(ages <= 1, 1, np.floor(ages))

# Head motion and trial count for each scan
hm = subject_data['HM'].values
trials = subject_data['trials'].values

# Region labels (short and full names)
reg_name = pd.read_csv(labels_file, header=None, delimiter='\t')[0]
reg_name_full = pd.read_csv(labels_full_file, header=None, delimiter='\t')[0]

# Number of scans (not unique participants) and regions
num_sub = len(subjects)
num_reg = len(reg_name)

# %% 2 - SpecParam (fooof-ing)
"""
Loads SpecParam Function and performs algorithm on PSD using chosen parameters
"""
import functions.fooof_func as fooof_func

importlib.reload(fooof_func)

from functions.fooof_func import run_specparam_batch, SpectralModel
fooof_settings = {
    'freq_range': [3, 40],  # or change to [4, 40] etc.
    'peak_width_limits': [2, 4], #specparam recommends min > half freq res
    'max_n_peaks': np.inf,
    'peak_threshold': 0,
    'min_peak_height': 0,
    'aperiodic_mode': 'fixed',
    'space': 'linear'  
}

fooof_results = run_specparam_batch(prime_psd, prime_freq, num_sub, num_reg, fooof_settings)

fooof_periodic_psd = fooof_results['periodic']
fooof_aperiodic_psd = fooof_results['aperiodic']
fooof_spectrum_psd = fooof_results['spectrum']
fooof_params = fooof_results['params']  
fooof_params[:,:,0] = 10**fooof_params[:,:,0] #converts params from log10 space 
fooof_rmse = fooof_results['r_squared']
fooof_models = fooof_results['models'] 
n_bins = fooof_results['n_bins']
freq_bins_low = fooof_results['freqs_used']

# fooof_params index layout (fixed mode): [0] = Offset, [1] = Slope
# (knee mode would be: [0] = Offset, [1] = Knee, [2] = Slope)
offset_mean = np.mean(fooof_params[:,:,0],axis = 1)
slope_mean = np.mean(fooof_params[:,:,-1],axis = 1)


    
# %% 3 - Properly Framing the Data 
"""
Creating Dataframes used in sections 4-6 and the band power arrays used in
sections 9-10
"""

periodic_adj = fooof_periodic_psd[:, 0:46, :38]  #Slices depends on your PSD Freq
tasks_arr = np.array(tasks)
be_idx = np.where(tasks_arr == 'babyeinstein')[0]
is_idx = np.where(tasks_arr == 'inscapes')[0]

# set Frequency 
freq_bins = prime_freq[6:138]
num_freqs= len(freq_bins)

#Re-Cutting the fooof freq bins for 3 to 30Hz 
freq_bins_low2 = freq_bins_low[0:46]
num_freqs_low = len(freq_bins_low2)

#Datafame containing all raw PSD
df_psd_all = pd.DataFrame({
    'Subject': np.repeat(subjects, num_freqs),
    'Age': np.repeat(ages, num_freqs),
    'Task' : np.repeat(tasks, num_freqs),
    'Age_bin':np.repeat(age_bin, num_freqs),
    'Frequency': np.tile(freq_bins,num_sub),
    'PSD': np.mean(prime_psd[:,6:138,:], axis = 2).flatten()
    })

# DataFrame contianing low frequency (3-30Hz) + fooof'd data
df_psd_low = pd.DataFrame({
    'Subject':np.repeat(subjects,num_freqs_low),
    'Age':np.repeat(ages,num_freqs_low),
    'Age_bin':np.repeat(age_bin,num_freqs_low),
    'Task' : np.repeat(tasks, num_freqs_low),
    'HM':np.repeat(hm,num_freqs_low),
    'Frequency':np.tile(freq_bins_low2,num_sub),
    'PSD_raw':np.mean(prime_psd[:,6:52,:], axis = 2).flatten(),
    'PSD':np.mean(periodic_adj,axis = 2).flatten(),
    'ap_PSD': np.mean(fooof_aperiodic_psd[:,0:46,:],axis = 2).flatten(),
    'Spectrum':np.mean(fooof_spectrum_psd[:,0:46,:],axis = 2).flatten()
    })


#Dataframe containing Slope + Offset per region
df_fooof_params_reg = pd.DataFrame({
    'Subject': np.tile(subjects,num_reg),
    'Subject_visit_ID':np.tile(subject_visitId,num_reg),
    'Sex': np.tile(prime_sex,num_reg),
    'Age': np.tile(ages,num_reg),
    'HM': np.tile(hm,num_reg),
    'Task' : np.tile(tasks, num_reg),
    'Region': np.repeat(reg_name, num_sub),
    'Slope': fooof_params[:,:,-1].T.flatten(),
    'Offset': fooof_params[:,:,0].T.flatten()
    })


# Flatten subject-region grid correctly
subjects_flat = np.tile(subjects, num_reg)              # Repeat each subject block for each region
subject_ids_flat = np.tile(subject_visitId, num_reg)
sex_flat = np.tile(prime_sex, num_reg) 
ages_flat = np.tile(ages, num_reg)
tasks_flat = np.tile(tasks, num_reg)
hm_flat = np.tile(hm, num_reg)
regions_flat = np.repeat(reg_name, num_sub)             # Repeat each region once per subject

# Construct dataframe properly for regional slope-offset
df_fooof_params_avg = pd.DataFrame({
    'Subject': subjects,
    'Age': ages,
    'Sex': prime_sex,
    'Age_bin':age_bin,
    'Task' : tasks,
    'Slope': slope_mean,
    'Offset': offset_mean
})


#Dataframes for psd stuff
"""
For NORMALIZED timeseries data, make sure you SUM the powers
For ZSCORE timeseries data, use MEAN
"""

#Dataframe for each ptp's region's avg psd for each band
theta_psd = np.mean(fooof_periodic_psd[:,0:5,:], axis = 1)
theta_psd_flat = theta_psd.T.flatten()
alpha_psd = np.mean(fooof_periodic_psd[:,5:17,:], axis = 1)
alpha_psd_flat = alpha_psd.T.flatten()
beta_psd = np.mean(fooof_periodic_psd[:,17:46,:], axis = 1)
beta_psd_flat = beta_psd.T.flatten()
lowgamma_psd = np.mean(prime_psd[:,52:94,:], axis = 1)
lowgamma_psd_flat = lowgamma_psd.T.flatten()
highgamma_psd = np.mean(prime_psd[:,111:137,:], axis = 1)
highgamma_psd_flat = highgamma_psd.T.flatten()

df_psd_reg_band = pd.DataFrame({
    'Subject': subjects_flat,
    'Subject_visit_ID': subject_ids_flat,
    'Region': regions_flat,
    'Age': ages_flat,
    'Sex':sex_flat,
    'Task': tasks_flat,
    'HM': hm_flat,
    'Theta': theta_psd_flat,
    'Alpha': alpha_psd_flat,
    'Beta': beta_psd_flat, 
    'LowGamma':lowgamma_psd_flat,
    'HighGamma':highgamma_psd_flat
    })


#Dataframe for each ptp's  raw region's avg psd for each band
theta_raw_psd = np.mean(prime_psd[:, 6:11, :], axis = 1)
theta_raw_psd_flat = theta_raw_psd.T.flatten()
alpha_raw_psd = np.mean(prime_psd[:, 11:23, :], axis = 1)
alpha_raw_psd_flat = alpha_raw_psd.T.flatten()
beta_raw_psd = np.mean(prime_psd[:, 23:52, :], axis = 1)
beta_raw_psd_flat = beta_raw_psd.T.flatten()
lowgamma_psd = np.mean(prime_psd[:,52:94,:], axis = 1)
lowgamma_psd_flat = lowgamma_psd.T.flatten()
highgamma_psd = np.mean(prime_psd[:,111:137,:], axis = 1)
highgamma_psd_flat = highgamma_psd.T.flatten()

df_raw_psd_reg_band = pd.DataFrame({
    'Subject': subjects_flat,
    'Subject_visit_ID': subject_ids_flat,
    'Region': regions_flat,
    'Age': ages_flat,
    'Sex': sex_flat,
    'HM': hm_flat,
    'Task': tasks_flat,
    'Theta': theta_raw_psd_flat,
    'Alpha': alpha_raw_psd_flat,
    'Beta': beta_raw_psd_flat,
    'LowGamma':lowgamma_psd_flat,
    'HighGamma':highgamma_psd_flat
    })




# %% 4 - Plotting Various PSDs
"""
CONFIG:
  band_param : 'low'   → fooof'd 3–30 Hz  (df_psd_low)
               'all'   → raw    3–80 Hz   (df_psd_all)
               'gamma' → raw   30–80 Hz   (df_psd_all slice)
  y_param    : for 'low'  → 'PSD' (periodic), 'Spectrum', 'ap_PSD'
               for 'all' / 'gamma' → 'PSD' only
"""
sns.set(style='white')
band_param  = 'low'    # 'all' | 'low' | 'gamma'
y_param     = 'ap_PSD' # for 'low': 'PSD' | 'Spectrum' | 'ap_PSD'
log         = True
save_fig    = False
show_legend = False

if band_param == 'all':
    df_plot, yval, xlims = df_psd_all, 'PSD', (3, 80)
elif band_param == 'low':
    df_plot, yval, xlims = df_psd_low, y_param, (3, 30)
else:
    df_plot = df_psd_all[df_psd_all['Frequency'].between(30, 80)]
    yval, xlims = 'PSD', (30, 80)

fig, ax = plt.subplots(figsize=(11.25, 10))
ax.set_xlim(*xlims)
shade_bands(ax, band_param)

sns.lineplot(df_plot, x="Frequency", y=yval,
             errorbar=('ci', 95), estimator='mean',
             hue='Task', hue_order=['babyeinstein', 'inscapes'],
             palette={'babyeinstein': 'tab:blue', 'inscapes': 'tab:orange'},
             linewidth=4)

if log:
    ax.set_yscale('log')
    logtxt = 'log'
else:
    logtxt = 'norm'

ax.set_xlabel("Frequency (Hz)", fontsize=70)
ax.set_ylabel(r'Power (Hz$^{-1}$)', fontsize=70)
ax.tick_params(axis='both', which='major', labelsize=70)

if not show_legend:
    ax.get_legend().remove()
else:
    ax.legend(fontsize=30, frameon=True,framealpha = 1)

plt.grid(False)
plt.tight_layout()

if save_fig:
    os.makedirs(os.path.join(base_dir, 'psd'), exist_ok=True)
    fname = f'{logtxt}scale_{band_param}_avg_{yval}{"_leg" if show_legend else ""}.png'
    plt.savefig(os.path.join(base_dir, 'psd', fname), dpi=150)

plt.show()


# %% 4.1 - Individual Participant PSDs
"""
Same options as section 4.
Plots each participant's spectrum, averaged across regions, for each task.
Group means can be overlaid to show individual variability around each task
average and help identify outliers.

Additional options:
  show_mean  : overlay per-task group mean on top of individual lines
  alpha_ind  : transparency of individual participant lines
"""

sns.set(style='white')

band_param = 'low'
y_param    = 'ap_PSD'
log        = True
show_mean  = True
alpha_ind  = 0.35
save_fig   = False

task_palette = {'babyeinstein': 'tab:blue', 'inscapes': 'tab:orange'}

# Select data (identical logic to section 4)
if band_param == 'low':
    df_plot = df_psd_low.copy()
    yval    = y_param
    xlims   = (3, 30)

elif band_param == 'all':
    df_plot = df_psd_all.copy()
    yval    = 'PSD'
    xlims   = (3, 80)

else:
    df_plot = df_psd_all[df_psd_all['Frequency'].between(30, 80)].copy()
    yval    = 'PSD'
    xlims   = (30, 80)


# Figure
fig, ax = plt.subplots(figsize=(11.25, 10))
ax.set_xlim(*xlims)

shade_bands(ax, band_param)

# Individual lines
for task, color in task_palette.items():
    task_df = df_plot[df_plot['Task'] == task]
    for _subj, subdf in task_df.groupby('Subject'):
        ax.plot(subdf['Frequency'].values, subdf[yval].values,
                color=color, alpha=alpha_ind, linewidth=1.2, zorder=2)

# Group mean lines on top
if show_mean:
    for task, color in task_palette.items():
        mean_df = (df_plot[df_plot['Task'] == task]
                   .groupby('Frequency', sort=True)[yval]
                   .mean()
                   .reset_index())
        ax.plot(mean_df['Frequency'], mean_df[yval],
                color=color, linewidth=4, zorder=5, label=task)

# Legend
legend_handles = [
    Line2D([0], [0], color=c, linewidth=3, label=t)
    for t, c in task_palette.items()
]
ax.legend(handles=legend_handles, fontsize=30, frameon=False)

if log:
    ax.set_yscale('log')
    logtxt = 'log'
else:
    logtxt = 'norm'

ax.set_xlabel("Frequency (Hz)", fontsize=70)
ax.set_ylabel(r'Power (Hz$^{-1}$)', fontsize=70)
ax.tick_params(axis='both', which='major', labelsize=70)
plt.grid(False)
plt.tight_layout()

if save_fig:
    os.makedirs(os.path.join(base_dir, 'psd'), exist_ok=True)
    filepath = os.path.join(base_dir, 'psd',
                            f'{logtxt}scale_{band_param}_ind_{yval}.png')
    plt.savefig(filepath, dpi=150)
plt.show()


# %% 5 - Paired t-test: Slope & Offset (BE vs IS)
"""
Compares whole-brain aperiodic slope and offset between Baby Einstein and
Inscapes using paired t-tests. Participants are matched across tasks, and
results are shown as paired scatter plots.
"""

save_fig = True

# Pivot to enforce subject matching
slope_wide = (
    df_fooof_params_avg[['Subject', 'Task', 'Slope']]
    .pivot(index='Subject', columns='Task', values='Slope')
    [['babyeinstein', 'inscapes']]
    .dropna()
)
offset_wide = (
    df_fooof_params_avg[['Subject', 'Task', 'Offset']]
    .pivot(index='Subject', columns='Task', values='Offset')
    [['babyeinstein', 'inscapes']]
    .dropna()
)

be_slope  = slope_wide['babyeinstein'].to_numpy()
is_slope  = slope_wide['inscapes'].to_numpy()
be_offset = offset_wide['babyeinstein'].to_numpy()
is_offset = offset_wide['inscapes'].to_numpy()

print(f"N (slope) = {len(be_slope)}, N (offset) = {len(be_offset)}")

t_slope,  p_slope  = ttest_rel(be_slope,  is_slope)
t_offset, p_offset = ttest_rel(be_offset, is_offset)

print(f"Slope:  t = {t_slope:.3f}, p = {p_slope:.4g}")
print(f"Offset: t = {t_offset:.3f}, p = {p_offset:.4g}")

os.makedirs(os.path.join(base_dir, 'psd'), exist_ok=True)

fig_slope = plot_paired_comparison(
    be_slope, is_slope,
    ylabel='Aperiodic Slope', title=f'($p = {p_slope:.3g}$)'
)
if save_fig:    
    fig_slope.savefig(os.path.join(base_dir, 'psd', 'BEvIS_Slope_ap_PSD.png'),
                      bbox_inches='tight', dpi=150)

fig_offset = plot_paired_comparison(
    be_offset, is_offset,
    ylabel='Aperiodic Offset', title=f'($p = {p_offset:.3g}$)'
)
if save_fig:
    fig_offset.savefig(os.path.join(base_dir, 'psd', 'BEvIS_Offset_ap_PSD.png'),
                       bbox_inches='tight', dpi=150)

# %% 5.1 - Regional Slope & Offset: Paired t-test (BE vs IS)
"""
Repeats the paired t-tests from section 5 in each region, with FDR correction
across regions.
"""

save_fig = True

# CONFIG
params     = ['Slope', 'Offset']
region_order = reg_name.tolist() 

all_results = {}  # stores results df per param, e.g. all_results['Slope']

for param in params:

    # Average over repeated visits per Subject × Task × Region 
    df = (df_fooof_params_reg
          .groupby(['Subject', 'Task', 'Region'], as_index=False)[param]
          .mean())

    #  Per-region paired t-test 
    be_mean = np.full(len(region_order), np.nan)
    is_mean = np.full(len(region_order), np.nan)
    ttest_t = np.full(len(region_order), np.nan)
    ttest_p = np.full(len(region_order), np.nan)

    for i, reg in enumerate(region_order):
        subdf = df[df['Region'] == reg]

        wide = subdf.pivot(index='Subject', columns='Task', values=param)

        if ('babyeinstein' not in wide.columns) or ('inscapes' not in wide.columns):
            continue

        paired = wide[['babyeinstein', 'inscapes']].dropna()
        if len(paired) < 2:
            continue

        be_vals = paired['babyeinstein'].to_numpy()
        is_vals = paired['inscapes'].to_numpy()

        be_mean[i] = np.nanmean(be_vals)
        is_mean[i] = np.nanmean(is_vals)

        t_stat, p_val = ttest_rel(be_vals, is_vals, nan_policy='omit')
        ttest_t[i] = t_stat
        ttest_p[i] = p_val

    # FDR correction over regions 
    valid = ~np.isnan(ttest_p)
    reject = np.zeros(len(region_order), dtype=bool)
    p_fdr  = np.full(len(region_order), np.nan)

    if valid.sum() > 0:
        rej_sub, p_fdr_sub, _, _ = multipletests(
            ttest_p[valid], alpha=0.05, method='fdr_bh'
        )
        reject[valid] = rej_sub
        p_fdr[valid]  = p_fdr_sub

    # Results DataFrame
    results = pd.DataFrame({
        'Region':       region_order,
        'mean_BE':      be_mean,
        'mean_IS':      is_mean,
        't_paired':     ttest_t,
        'p_paired':     ttest_p,
        'p_FDR':        p_fdr,
        'sig_FDR_0.05': reject,
    })

    all_results[param] = results

    # T-maps for brain plotting
    # t_map_raw : uncorrected threshold (p <= 0.05)
    # t_map_fdr : FDR-corrected threshold (q <= 0.05)
    results['t_map_raw'] = np.where(
        (ttest_p <= 0.05) & ~np.isnan(ttest_p), ttest_t, 0
    )
    results['t_map_fdr'] = np.where(reject, ttest_t, 0)

    print(f"\n=== {param} ===")
    print(results[['Region', 't_paired', 'p_paired', 'p_FDR', 'sig_FDR_0.05']]
          .sort_values('p_paired')
          .to_string(index=False))



# %% 5.2 - Brain Plot Slope/Offset and t-stat 
"""
Brain plots of slope-offset t-test results from above
"""
import importlib
import functions #<--- Re-imports if using Spyder, as it loses loads sometimes
import functions.brainplots as brainplots
importlib.reload(functions)
importlib.reload(brainplots)

os.makedirs(os.path.join(base_dir, 'slopeoff'), exist_ok=True)

for param, results in all_results.items():

    be_mean  = results['mean_BE'].to_numpy()
    is_mean  = results['mean_IS'].to_numpy()
    ttest_t  = results['t_paired'].to_numpy()
    t_map_fdr = results['t_map_fdr'].to_numpy()

    # Shared colour scale for BE / IS maps (zero-referenced)
    be_norm = be_mean - np.nanmin(be_mean)
    is_norm = is_mean - np.nanmin(is_mean)

    apdatmin  = np.nanmin(np.hstack([be_norm, is_norm]))
    apdatmax  = np.nanmax(np.hstack([be_norm, is_norm]))
    labapdatmin = np.nanmin(np.hstack([be_mean, is_mean]))
    labapdatmax = np.nanmax(np.hstack([be_mean, is_mean]))

    tmax = np.nanmax(np.abs(t_map_fdr))

    # BE brain plot
    fig1 = brainplots.surface_brain_plotv2(
        make_4d_atlas_nifti(image.load_img(atlasimg), be_norm),
        FS_DIR, surf='pial', fade=True,
        cmap='BuGn', symmetric=False,
        cbar_label=fr'Aperiodic {param} BE',
        datmin=apdatmin, datmax=apdatmax,
        labdatmin=labapdatmin, labdatmax=labapdatmax,
        setmin=True, setmax=True,sym_div = 10
    )
    fig1.savefig(os.path.join(base_dir, 'slopeoff', f'BE_{param}_bp.png'),
                 bbox_inches='tight', dpi=150)
    plt.close(fig1)

    # IS brain plot
    fig2 = brainplots.surface_brain_plotv2(
        make_4d_atlas_nifti(image.load_img(atlasimg), is_norm),
        FS_DIR, surf='pial', fade=True,
        cmap='BuGn', symmetric=False,
        cbar_label=fr'Aperiodic {param} IS',
        datmin=apdatmin, datmax=apdatmax,
        labdatmin=labapdatmin, labdatmax=labapdatmax,
        setmin=True, setmax=True,sym_div = 10
    )
    fig2.savefig(os.path.join(base_dir, 'slopeoff', f'IS_{param}_bp.png'),
                 bbox_inches='tight', dpi=150)
    plt.close(fig2)

    # t-stat brain plot (FDR-thresholded)
    fig3 = brainplots.surface_brain_plotv2(
        make_4d_atlas_nifti(image.load_img(atlasimg), t_map_fdr),
        FS_DIR, surf='pial', fade=True,
        cmap='coolwarm', symmetric=True,
        cbar_label=fr'Aperiodic {param} $t$',
        datmin=-tmax, datmax=tmax,sym_div = 2
    )
    fig3.savefig(os.path.join(base_dir, 'slopeoff', f'BEvIS_{param}_t.png'),
                 bbox_inches='tight', dpi=150)
    plt.close(fig3)

    print(f"Saved brain plots for {param}")


# %% 6 - Regional Band Power: Paired t-test (BE vs IS)
"""
Compares band power between Baby Einstein and Inscapes in each region using
paired t-tests with FDR correction.

df_source : df_psd_reg_band     -> FOOOF'd theta, alpha, and beta; raw gamma
            df_raw_psd_reg_band -> raw power for all bands
"""

# CONFIG
df_source = df_psd_reg_band   # swap to df_raw_psd_reg_band for unfooof'd
source_label = 'raw' if df_source is df_raw_psd_reg_band else 'fooof'
save_fig  = True

band_cols = ['Theta', 'Alpha', 'Beta', 'LowGamma', 'HighGamma']
regions   = list(reg_name)   # stable atlas order from section 1
num_reg   = len(regions)

os.makedirs(os.path.join(base_dir, 'psd_bands'), exist_ok=True)



# Prep: normalise Task strings, collapse repeated visits
df = df_source.copy()
df['Task'] = df['Task'].astype(str).str.strip().str.lower()
df = (df
      .groupby(['Subject', 'Task', 'Region'], as_index=False)[band_cols]
      .mean())

# Per-region paired t-tests
# Results stored in a dict keyed by band for clean downstream access
band_results = {}

for band in band_cols:
    tstat = np.full(num_reg, np.nan)
    pval  = np.full(num_reg, np.nan)

    for i, reg in enumerate(regions):
        tstat[i], pval[i] = paired_t_for_region(df, reg, band)

    pval_fdr, reject = fdr_nan_safe(pval)

    tstat_masked = tstat.copy()
    tstat_masked[~reject] = 0

    band_results[band] = {
        'tstat':        tstat,
        'pval':         pval,
        'pval_fdr':     pval_fdr,
        'reject':       reject,
        'tstat_masked': tstat_masked,
    }

    print(f"\nTop regions — {band}:")
    print(top_table(pval, tstat, regions).to_string(index=False))


# %% 6.1 - Brain Plots: Regional Band t-stats
"""
Plots the FDR-corrected regional t-stats from section 6 on the brain surface.
Currently set to plot beta only.
"""
import importlib
import functions.brainplots as brainplots
importlib.reload(brainplots)

band_labels = {
    'Theta':     r'$\theta$',
    'Alpha':     r'$\alpha$',
    'Beta':      r'$\beta$',
    'LowGamma':  r'$\gamma_1$',
    'HighGamma': r'$\gamma_2$',
}

band_filenames = {
    'Theta':     'theta',
    'Alpha':     'alpha',
    'Beta':      'beta',
    'LowGamma':  'lowgamma',
    'HighGamma': 'highgamma',
}

for band, res in band_results.items():
    if band == 'Beta':
        dat  = res['tstat_masked']
        tmax = np.nanmax(np.abs(dat)) if np.any(dat != 0) else 0
    
        fig = brainplots.surface_brain_plotv2(
            make_4d_atlas_nifti(image.load_img(atlasimg), dat),
            r'FS_DIR', surf='pial', fade=True, cort=(0.9, 0.9, 0.9),
            cmap='coolwarm', symmetric=True,
            cbar_label=fr'{band_labels[band]} t-stat',
            datmin=-tmax, datmax=tmax,sym_div = 10
        )
    
        if save_fig:
            fig.savefig(
                os.path.join(base_dir, 'psd_bands', f'{band_filenames[band]}_{source_label}_t_bp.png'),
                dpi=150, bbox_inches='tight'
            )

    #plt.close(fig)


# %% 6.2 - Whole-Brain Band Power: Paired t-test + Scatter Plots
"""
Repeats the paired t-tests on whole-brain band power, averaged across regions,
and shows the results as paired scatter plots.
"""
out_dir = os.path.join(base_dir, 'psd_bands', 'wholebrain')
os.makedirs(out_dir, exist_ok=True)

# Whole-brain mean: average over all regions per Subject × Task
wb = (df
      .groupby(['Subject', 'Task'], as_index=False)[band_cols]
      .mean())

# Run whole-brain tests + plots
wholebrain_stats = []

for band in band_cols:
    be_vals, is_vals, t_wb, p_wb = paired_t_wholebrain(wb, band)

    wholebrain_stats.append({
        'Band': band,
        't':    t_wb,
        'p':    p_wb,
        'n_BE': int(np.sum(~np.isnan(be_vals))),
        'n_IS': int(np.sum(~np.isnan(is_vals))),
    })

    fig = plot_paired_comparison(
        be_vals, is_vals,
        ylabel=band,
        title=f'($p = {p_wb:.3g}$)'
    )

    if save_fig:
        fig.savefig(
            os.path.join(out_dir, f'WholeBrain_{band}.png'),
            dpi=150, bbox_inches='tight'
        )
    plt.close(fig)

wholebrain_stats_df = pd.DataFrame(wholebrain_stats).sort_values('p', na_position='last')
print(wholebrain_stats_df)

# %% 6.3 - Compute Per-Subject Band Arrays (BE vs IS)
"""
Computes each participant's band power by task for the brain plots in section 6.3.1.
"""

from functions.brainplots import make_4d_atlas_nifti, surface_brain_plotv2
from nilearn import image

# Derive task indices from labels (safe against reordering)
tasks_arr = np.array(tasks)
be_idx = np.where(tasks_arr == 'babyeinstein')[0]
is_idx = np.where(tasks_arr == 'inscapes')[0]

# Fooof periodic bands
fooof_be = {
    'theta'  : np.mean(fooof_periodic_psd[be_idx,  0:5,  :], axis=1),
    'alpha'  : np.mean(fooof_periodic_psd[be_idx,  5:17, :], axis=1),
    'beta'   : np.mean(fooof_periodic_psd[be_idx, 17:46, :], axis=1),
    'lo_beta': np.mean(fooof_periodic_psd[be_idx, 17:30, :], axis=1),
    'hi_beta': np.mean(fooof_periodic_psd[be_idx, 30:46, :], axis=1),
}

fooof_is = {
    'theta'  : np.mean(fooof_periodic_psd[is_idx,  0:5,  :], axis=1),
    'alpha'  : np.mean(fooof_periodic_psd[is_idx,  5:17, :], axis=1),
    'beta'   : np.mean(fooof_periodic_psd[is_idx, 17:46, :], axis=1),
    'lo_beta': np.mean(fooof_periodic_psd[is_idx, 17:30, :], axis=1),
    'hi_beta': np.mean(fooof_periodic_psd[is_idx, 30:46, :], axis=1),
}

# Raw PSD bands
raw_be = {
    'theta'    : np.mean(prime_psd[be_idx,   6:11,  :], axis=1),
    'alpha'    : np.mean(prime_psd[be_idx,  11:23,  :], axis=1),
    'beta'     : np.mean(prime_psd[be_idx,  23:52,  :], axis=1),
    'lo_beta'  : np.mean(prime_psd[be_idx,  23:36,  :], axis=1),
    'hi_beta'  : np.mean(prime_psd[be_idx,  36:52,  :], axis=1),
    'lowgamma' : np.mean(prime_psd[be_idx,  52:94,  :], axis=1),
    'highgamma': np.mean(prime_psd[be_idx, 111:137, :], axis=1),
}

raw_is = {
    'theta'    : np.mean(prime_psd[is_idx,   6:11,  :], axis=1),
    'alpha'    : np.mean(prime_psd[is_idx,  11:23,  :], axis=1),
    'beta'     : np.mean(prime_psd[is_idx,  23:52,  :], axis=1),
    'lo_beta'  : np.mean(prime_psd[is_idx,  23:36,  :], axis=1),
    'hi_beta'  : np.mean(prime_psd[is_idx,  36:52,  :], axis=1),
    'lowgamma' : np.mean(prime_psd[is_idx,  52:94,  :], axis=1),
    'highgamma': np.mean(prime_psd[is_idx, 111:137, :], axis=1),
}

# Lookup dicts (colour and latex label per band)
band_cmap = {
    'theta'    : 'RdPu',
    'alpha'    : 'Purples',
    'beta'     : 'Greens',
    'lo_beta'  : 'Greens',
    'hi_beta'  : 'Greens',
    'lowgamma' : 'Oranges',
    'highgamma': 'Reds',
}

band_label = {
    'theta'    : r'\theta',
    'alpha'    : r'\alpha',
    'beta'     : r'\beta',
    'lo_beta'  : r'\beta_1',
    'hi_beta'  : r'\beta_2',
    'lowgamma' : r'\gamma_1',
    'highgamma': r'\gamma_2',
}

# Load atlas image once
atlas_img = image.load_img(atlasimg)


# %% 6.3.1 - Brain Plot Band Power Maps (BE and IS)
"""
Plots mean regional band power for each task on the brain surface, using a
  shared colour scale.
"""
import importlib
import functions.brainplots as brainplots
importlib.reload(brainplots)

# CONFIG
band   = 'beta'    # 'theta','alpha','beta','lo_beta','hi_beta','lowgamma','highgamma'
source = 'fooof'      # 'raw' or 'fooof'
save_fig = False

os.makedirs(os.path.join(base_dir, 'psd_bands'), exist_ok=True)

def get_band_arrays(band, source='raw'):
    """Return (psd_ind_be, psd_ind_is) for the requested band and source."""
    if source == 'raw':
        be_map, is_map = raw_be, raw_is
    elif source == 'fooof':
        be_map, is_map = fooof_be, fooof_is
    else:
        raise ValueError("source must be 'raw' or 'fooof'")
    if band not in be_map:
        raise ValueError(
            f"Band '{band}' not available for source '{source}'. "
            f"Available: {list(be_map.keys())}"
        )
    return be_map[band], is_map[band]


def plot_band_maps(band, source='raw', save_fig=False,
                   fs_path=FS_DIR):
    """
    Plot BE and IS brain maps for a given band and source.
    Colour scale is shared across both maps for direct comparison.
    """
    psd_ind_be, psd_ind_is = get_band_arrays(band, source)

    psd_be = np.mean(psd_ind_be, axis=0)
    psd_is = np.mean(psd_ind_is, axis=0)

    # Shared colour scale 
    global_min = np.min(np.hstack([psd_be, psd_is]))
    global_max = np.max(np.hstack([psd_be, psd_is]))

    psd_be_shift = psd_be - global_min
    psd_is_shift = psd_is - global_min

    apdatmin = 0.0
    apdatmax = np.max(np.hstack([psd_be_shift, psd_is_shift]))

    cmap  = band_cmap[band]
    blabel = band_label[band]

    def _make_fig(nifti, task_str):
        return brainplots.surface_brain_plotv2(
            nifti, fs_path,
            surf='pial', fade=True,
            setmax=True, setmin=True,
            cmap=cmap, symmetric=False,
            cbar_label=fr'${blabel}$ {task_str} (Hz$^{{-1}}$)',
            datmin=apdatmin, datmax=apdatmax,
            labdatmin=global_min, labdatmax=global_max,sym_div = 8
        )

    fig_be = _make_fig(make_4d_atlas_nifti(atlas_img, psd_be_shift), 'BE')
    fig_is = _make_fig(make_4d_atlas_nifti(atlas_img, psd_is_shift), 'IS')

    if save_fig:
        out_dir = os.path.join(base_dir, 'psd_bands')
        fig_be.savefig(
            os.path.join(out_dir, f'{band}_{source}_BE.png'),
            dpi=150, bbox_inches='tight'
        )
        fig_is.savefig(
            os.path.join(out_dir, f'{band}_{source}_IS.png'),
            dpi=150, bbox_inches='tight'
        )
    # plt.close(fig_be)
    # plt.close(fig_is)

    return fig_be, fig_is


# Single band call
fig_be, fig_is = plot_band_maps(band, source, save_fig=save_fig)

# Or loop all bands
# for band in band_cmap:
#     plot_band_maps(band, source='raw', save_fig=True)
# for band in band_cmap:
#     plot_band_maps(band, source='fooof', save_fig=True)

# %% 7 - Alpha Peak Analysis (BE vs IS)
"""
Finds each participant's dominant alpha peak, then compares peak frequency and
  amplitude between tasks. Regional comparisons use peak amplitude only.
"""
from scipy.signal import find_peaks
import importlib
import functions.brainplots as brainplots
from scipy.stats import ttest_rel
importlib.reload(brainplots)

atlas_img = image.load_img(atlasimg3d)

# CONFIG
alpha_slice = slice(2, 18)           # indices into fooof_periodic_psd freq axis
freq_band   = freq_bins_low[2:18]    # Shifted for wider envelope
save_fig    = False

band_psd = fooof_periodic_psd[:, alpha_slice, :]  # (num_sub, n_alpha_freqs, num_reg)

# Helper: dominant peak for a 1D spectrum
def dominant_peak(y, freqs, min_prominence=0.00001, min_distance_bins=2):
    """Returns (peak_freq, peak_amp) or (nan, nan) if no peak found."""
    y_filled = np.where(np.isnan(y), -np.inf, y)
    peaks, props = find_peaks(y_filled, prominence=min_prominence,
                              distance=min_distance_bins)
    if peaks.size == 0:
        return np.nan, np.nan
    best = peaks[np.argmax(props['prominences'])]
    return float(freqs[best]), float(y[best])

# Whole-brain peak: average regions first, then find peak per subject
wb_psd = np.mean(band_psd, axis=2)   # (num_sub, n_alpha_freqs)

wb_peak_freq = np.full(num_sub, np.nan)
wb_peak_amp  = np.full(num_sub, np.nan)

for i in range(num_sub):
    wb_peak_freq[i], wb_peak_amp[i] = dominant_peak(wb_psd[i], freq_band)

# Per-region peak
reg_peak_freq = np.full((num_sub, num_reg), np.nan)
reg_peak_amp  = np.full((num_sub, num_reg), np.nan)

for i in range(num_sub):
    for r in range(num_reg):
        reg_peak_freq[i, r], reg_peak_amp[i, r] = dominant_peak(
            band_psd[i, :, r], freq_band
        )

# Paired t-tests (BE vs IS) via pivot
def paired_peak_test(peak_arr, label):
    """Paired t-test on a whole-brain scalar peak metric."""
    df_tmp = pd.DataFrame({'Subject': subjects, 'Task': tasks, 'Value': peak_arr})
    wide   = df_tmp.pivot(index='Subject', columns='Task', values='Value')
    if ('babyeinstein' not in wide.columns) or ('inscapes' not in wide.columns):
        print(f"{label}: missing task column"); return np.nan, np.nan, None, None
    paired = wide[['babyeinstein', 'inscapes']].dropna()
    be_v   = paired['babyeinstein'].to_numpy()
    is_v   = paired['inscapes'].to_numpy()
    t, p   = ttest_rel(be_v, is_v, nan_policy='omit')
    print(f"{label}: t = {t:.3f}, p = {p:.4g}, n_pairs = {len(paired)}")
    return t, p, be_v, is_v

print("Whole-Brain Alpha Peak")
t_freq, p_freq, be_wb_freq, is_wb_freq = paired_peak_test(wb_peak_freq, 'Peak Freq')
t_amp,  p_amp,  be_wb_amp,  is_wb_amp  = paired_peak_test(wb_peak_amp,  'Peak Amp')

# Scatter plots (reuses plot_paired_comparison from Funcs)
fig_freq = plot_paired_comparison(
    be_wb_freq, is_wb_freq,
    ylabel=r'Peak $\alpha$ Freq (Hz)',
    title=f'($p = {p_freq:.3g}$)'
)
if save_fig:
    fig_freq.savefig(os.path.join(base_dir, 'psd_bands', 'alpha_peak_freq_wb.png'),
                     dpi=150, bbox_inches='tight')
plt.close(fig_freq)

fig_amp = plot_paired_comparison(
    be_wb_amp, is_wb_amp,
    ylabel=r'Peak $\alpha$ Amp (Hz$^{-1}$)',
    title=f'($p = {p_amp:.3g}$)'
)
if save_fig:
    fig_amp.savefig(os.path.join(base_dir, 'psd_bands', 'alpha_peak_amp_wb.png'),
                    dpi=150, bbox_inches='tight')
plt.close(fig_amp)

# Spectrum plot: whole-brain mean  95% CI by task
# Average over regions first, then over subjects within each task
be_spec = np.nanmean(np.mean(band_psd[be_idx], axis=2), axis=0)  # (n_alpha_freqs,)
is_spec = np.nanmean(np.mean(band_psd[is_idx], axis=2), axis=0)

be_ci = 1.96 * np.nanstd(np.mean(band_psd[be_idx], axis=2), axis=0) / np.sqrt(len(be_idx))
is_ci = 1.96 * np.nanstd(np.mean(band_psd[is_idx], axis=2), axis=0) / np.sqrt(len(is_idx))

fig_spec, ax_spec = plt.subplots(figsize=(11.25, 10))
ax_spec.plot(freq_band, be_spec, color='tab:blue',   linewidth=4, label='BE')
ax_spec.fill_between(freq_band, be_spec - be_ci, be_spec + be_ci,
                     alpha=0.3, color='tab:blue')
ax_spec.plot(freq_band, is_spec, color='tab:orange', linewidth=4, label='IS')
ax_spec.fill_between(freq_band, is_spec - is_ci, is_spec + is_ci,
                     alpha=0.3, color='tab:orange')
ax_spec.set_xlabel('Frequency (Hz)', fontsize=70)
ax_spec.set_ylabel(r'Power (Hz$^{-1}$)', fontsize=70)
ax_spec.tick_params(axis='both', labelsize=70)
ax_spec.set_xlim(freq_band[0], freq_band[-1])
ax_spec.legend(fontsize=40, frameon=False)
ax_spec.spines['top'].set_visible(False)
ax_spec.spines['right'].set_visible(False)
plt.tight_layout()
if save_fig:
    fig_spec.savefig(os.path.join(base_dir, 'psd_bands', 'alpha_peak_spectrum.png'),
                     dpi=150, bbox_inches='tight')
plt.show()

# Per-region paired t-test on peak amplitude - brain plot
reg_amp_tstat = np.full(num_reg, np.nan)
reg_amp_pval  = np.full(num_reg, np.nan)

for r, reg in enumerate(list(reg_name)):
    df_tmp = pd.DataFrame({
        'Subject': subjects,
        'Task':    tasks,
        'Value':   reg_peak_amp[:, r]
    })
    wide   = df_tmp.pivot(index='Subject', columns='Task', values='Value')
    if ('babyeinstein' not in wide.columns) or ('inscapes' not in wide.columns):
        continue
    paired = wide[['babyeinstein', 'inscapes']].dropna()
    if len(paired) < 2:
        continue
    t, p = ttest_rel(paired['babyeinstein'].values,
                     paired['inscapes'].values, nan_policy='omit')
    reg_amp_tstat[r] = t
    reg_amp_pval[r]  = p

reg_amp_pval_fdr, reg_amp_reject = fdr_nan_safe(reg_amp_pval)
reg_amp_tstat_masked = np.where(reg_amp_reject, reg_amp_tstat, 0)

tmax = np.nanmax(np.abs(reg_amp_tstat_masked)) if np.any(reg_amp_reject) else 1.0
fig_bp = brainplots.surface_brain_plotv2(
    make_atlas_nifti(atlas_img, reg_amp_tstat_masked),
    r'C:\mne\fs', surf='pial', fade=True, cort=(0.9, 0.9, 0.9),
    cmap='coolwarm', symmetric=True,
    cbar_label=r'$\alpha$ peak amp $t$-stat (FDR)',
    datmin=-tmax, datmax=tmax
)
if save_fig:
    fig_bp.savefig(os.path.join(base_dir, 'psd_bands', 'alpha_peak_amp_t_bp.png'),
                   dpi=150, bbox_inches='tight')
    
# QC: per-subject whole-brain spectra + detected peak
ncols = 6
nrows = int(np.ceil(num_sub / ncols))
fig_qc, axes = plt.subplots(nrows, ncols,
                            figsize=(2.6 * ncols, 2.2 * nrows),
                            sharex=True, constrained_layout=True)
axes = np.atleast_1d(axes).ravel()

for i in range(num_sub):
    ax  = axes[i]
    lbl = 'BE' if tasks[i] == 'babyeinstein' else 'IS'
    col = 'tab:blue' if tasks[i] == 'babyeinstein' else 'tab:orange'

    ax.plot(freq_band, wb_psd[i], color=col, lw=1.5)

    if np.isnan(wb_peak_freq[i]):
        ttl = f'{subjects[i]} | {lbl} | no peak'
    else:
        ax.plot(wb_peak_freq[i], wb_peak_amp[i], 'r*', ms=12, zorder=5)
        ax.axvline(wb_peak_freq[i], color='r', ls=':', lw=0.8, alpha=0.6)
        ttl = f'{subjects[i]} | {lbl} | {wb_peak_freq[i]:.1f} Hz'

    ax.set_title(ttl, fontsize=7)
    ax.tick_params(labelsize=6)

for j in range(num_sub, len(axes)):   # hide empty panels
    axes[j].axis('off')

fig_qc.supxlabel('Frequency (Hz)')
fig_qc.supylabel(r'Power (Hz$^{-1}$)')
fig_qc.suptitle('Whole-brain alpha spectra + detected peak', fontsize=12)
plt.show()

# QC: all whole-brain spectra on one axis + detected peaks
fig_qc, ax = plt.subplots(figsize=(10, 7))

for i in range(num_sub):
    col = 'tab:blue' if tasks[i] == 'babyeinstein' else 'tab:orange'
    ax.plot(freq_band, wb_psd[i], color=col, lw=0.8, alpha=0.4)
    if not np.isnan(wb_peak_freq[i]):
        ax.plot(wb_peak_freq[i], wb_peak_amp[i], '*', color='r',
                ms=10, alpha=0.7, zorder=5)

# legend proxies 
ax.legend(handles=[
    Line2D([0], [0], color='tab:blue',   lw=2, label='BE'),
    Line2D([0], [0], color='tab:orange', lw=2, label='IS'),
    Line2D([0], [0], color='r', marker='*', ls='', label='peak'),
], frameon=False)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel(r'Power (Hz$^{-1}$)')
ax.set_xlim(freq_band[0], freq_band[-1])
ax.set_title('All whole-brain alpha spectra + detected peaks')
plt.tight_layout()
plt.show()

# %% 8 - Age Related Changes
"""
Tests age effects on the selected whole-brain marker using a mixed model with a
  Task x Age interaction and a random intercept for each participant.
"""
import statsmodels.formula.api as smf
import patsy
from matplotlib.ticker import MaxNLocator

# Build sheets 
# Note: theta/alpha/beta use fooof periodic; lowgamma/highgamma use raw PSD
sheets = {
    'slope':          build_marker_df('slope',     slope_mean,
                                      fooof_params[:, :, -1]),
    'offset':         build_marker_df('offset',    offset_mean,
                                      fooof_params[:, :,  0]),
    'theta':          build_marker_df('theta',     np.mean(theta_psd,     axis=1),
                                      theta_psd),
    'alpha':          build_marker_df('alpha',     np.mean(alpha_psd,     axis=1),
                                      alpha_psd),
    'beta':           build_marker_df('beta',      np.mean(beta_psd,      axis=1),
                                      beta_psd),
    'lowgamma':       build_marker_df('lowgamma',  np.mean(lowgamma_psd,  axis=1),
                                      lowgamma_psd),
    'highgamma':      build_marker_df('highgamma', np.mean(highgamma_psd, axis=1),
                                      highgamma_psd),
    'alpha_peak_freq': build_marker_df('alpha_peak_freq', wb_peak_freq,
                                       reg_peak_freq, valid_range=(6, 13)),
    'alpha_peak_amp':  build_marker_df('alpha_peak_amp',  wb_peak_amp,
                                      reg_peak_amp),
}

param = 'beta'
df = sheets[param]
paramtitle = 'Beta'


def fit_interaction(df, y_col, group_col='Subject'):
    d = df.dropna(subset=[y_col, 'Task', 'Age']).copy()
    res = smf.mixedlm(
        f"Q('{y_col}') ~ C(Task) * center(Age)",
        d, groups=d[group_col]
    ).fit(reml=True)
    print(res.summary())
    return res


# Publication figure settings
PUB = dict(point_size=150, ylab=75, xlab=75, xtick=65, ytick=65, legend=11)

TASK_COLORS = {"babyeinstein": "#5B8DB8", "inscapes": "#E07B54"}
TASK_LABELS = {"babyeinstein": "Baby Einstein", "inscapes": "Inscapes"}

def _resolve(task, mapping, default):
    for k, v in mapping.items():
        if str(task).lower() == k.lower():
            return v
    return default



def plot_age_effects_pub(res, df, y_col, x_col='Age', task_col='Task',
                         param_lab=None, show_legend=False,
                         task_colors=TASK_COLORS, task_labels=TASK_LABELS,
                         back_transform=None, n_grid=50, ax=None,
                         sci_decimals=1, sci_limits=(-2, 3)):
    """
    Frequentist analogue of the brms/ggdist plotter.
    Fitted population-mean age effect per task (re_formula=NA equivalent),
    with nested 50% & 95% CI ribbons over the raw points.

    res            : fitted MixedLMResults; formula must use center(Age) & C(Task),
                     e.g. "y ~ C(Task)*center(Age)".
    back_transform : optional fn applied to predictions + data, e.g. for an
                     unlog: lambda v: 10**(v*log_sd + log_mean).
    """
    param_lab = param_lab or y_col
    bt = back_transform or (lambda v: v)

    d = df.dropna(subset=[y_col, x_col, task_col]).copy()
    design_info = res.model.data.design_info     # patsy spec from the fit
    fe_names = res.fe_params.index
    beta     = res.fe_params.values
    cov_fe   = res.cov_params().loc[fe_names, fe_names].values

    tasks    = list(d[task_col].astype(str).unique())
    age_grid = np.linspace(d[x_col].min(), d[x_col].max(), n_grid)

    new_ax = ax is None
    if new_ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    widths = [0.95, 0.50]

    for task in tasks:
        colour  = _resolve(task, task_colors, "#888888")
        grid_df = pd.DataFrame({x_col: age_grid, task_col: task})

        Xdf = patsy.build_design_matrices([design_info], grid_df,
                                          return_type='dataframe')[0]
        X   = Xdf.reindex(columns=fe_names).values          
        Xb  = X @ beta
        se  = np.sqrt(np.einsum('ij,jk,ik->i', X, cov_fe, X))

        for cov in widths:
            z = stats.norm.ppf(0.5 + cov / 2)
            ax.fill_between(age_grid, bt(Xb - z*se), bt(Xb + z*se),
                            color=colour, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(age_grid, bt(Xb), color=colour, linewidth=5, zorder=3)

        sub = d[d[task_col].astype(str) == task]
        ax.scatter(sub[x_col], bt(sub[y_col].values), color=colour,
                   alpha=0.75, s=PUB['point_size'], edgecolors=colour,
                   zorder=2, label=_resolve(task, task_labels, task))

    # theme_classic
    ax.set_xlabel("Age (years)", fontsize=PUB['xlab'])
    ax.set_ylabel(param_lab,     fontsize=PUB['ylab'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.6); ax.spines[s].set_color('black')
    ax.tick_params(colors='black', width=0.6, labelsize=PUB['xtick'])
    

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FixedSciFormatter(sci_decimals, sci_limits))
    ax.yaxis.get_offset_text().set_fontsize(40)   # match tick label size
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    # scale_y_continuous
    ymin, ymax = ax.get_ylim(); yr = ymax - ymin
    ax.set_ylim(-0.0001, ymax + 0.12*yr)#ymin - 0.01*yr
    ax.set_xlim(0.9, 6)

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.16),
                  frameon=False, ncol=len(tasks),
                  fontsize=PUB['legend'], title="Task")
    if new_ax:
        fig.tight_layout()
    return ax

res = fit_interaction(df, f'WholeBrain_{param}')
plot_age_effects_pub(res, df, f'WholeBrain_{param}', param_lab=paramtitle)

term = 'C(Task)[T.inscapes]:center(Age)'
coef = res.params[term]
pval = res.pvalues[term]

tag   = f'coef_{coef:+.3f}_p_{pval:.3f}'.replace('.', '-')
fname = f'{param}_{tag}.png'

lme_dir = os.path.join(base_dir, 'lme')
#plt.savefig(os.path.join(lme_dir, fname), dpi=150, bbox_inches='tight')
plt.show()

# %% 9 - Regional Age Related Changes (per-region table, all markers)
"""
 Repeats section 8 for every region and marker, and fits age effects within each
  task. 
  Results are FDR-corrected across regions and saved to Excel. 
  Positive interaction values mean a more positive age slope in Inscapes. 
  Int_flag lists any mixed model warnings.
"""
import warnings
import statsmodels.formula.api as smf

# CONFIG
save_xlsx   = True
age_alpha   = 0.05
age_min_n   = 5                     
AGE_BE      = 'babyeinstein'        
AGE_IS      = 'inscapes'
AGE_INT_TERM = f'C(Task)[T.{AGE_IS}]:center(Age)'

lme_dir = r'Y:\projects\OPM-Analysis\OPM_BE_vs_IS\analysis\lme'


def _age_ols(d_task):
    """marker ~ Age within one task. Returns (coef, p, r2, n)."""
    d_task = d_task.dropna(subset=['y', 'Age'])
    n = len(d_task)
    if n < age_min_n or d_task['Age'].nunique() < 3:
        return np.nan, np.nan, np.nan, n
    fit = smf.ols('y ~ Age', d_task).fit()
    return (float(fit.params['Age']), float(fit.pvalues['Age']),
            float(fit.rsquared), n)


def _age_interaction(d):
    """
    Mixed model from section 8 (fit_interaction), without the printed summary.
    Returns (coef, stat, p, n_subjects, flag) for the Task x Age term.
    """
    d = d.dropna(subset=['y', 'Age', 'Task'])
    n_subj = d['Subject'].nunique()
    n_per_task = d['Task'].value_counts()
    if (n_per_task.reindex([AGE_BE, AGE_IS]).fillna(0) < age_min_n).any():
        return np.nan, np.nan, np.nan, n_subj, 'too few sessions'


    y_sd = float(d['y'].std())
    if not np.isfinite(y_sd) or y_sd == 0:
        return np.nan, np.nan, np.nan, n_subj, 'no variance'
    d = d.assign(y=d['y'] / y_sd)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            fit = smf.mixedlm('y ~ C(Task) * center(Age)', d,
                              groups=d['Subject']).fit(reml=True)
        except Exception as err:                      # failed fit
            return (np.nan, np.nan, np.nan, n_subj,
                    f'fit failed: {type(err).__name__}')

    if AGE_INT_TERM not in fit.params.index:
        return np.nan, np.nan, np.nan, n_subj, 'interaction term not found'

    notes = sorted({str(w.message).split('.')[0] for w in caught})
    if not fit.converged:
        notes.insert(0, 'not converged')
    return (float(fit.params[AGE_INT_TERM]) * y_sd,
            float(fit.tvalues[AGE_INT_TERM]),
            float(fit.pvalues[AGE_INT_TERM]), n_subj, '; '.join(notes))


def regional_age_table(df_marker):
    """Build the per-region age table for one marker sheet."""
    # `sheets` dict from section 9 or section 10.
    wb_col = [c for c in df_marker.columns if c.startswith('WholeBrain_')][0]
    suffix = wb_col[len('WholeBrain_'):]

    rows = []
    for reg, reg_full in zip(reg_name, reg_name_full):
        d = (df_marker[['Subject', 'Task', 'Age', f'{reg}_{suffix}']]
             .rename(columns={f'{reg}_{suffix}': 'y'}))
        d['Task'] = d['Task'].astype(str)

        be_coef, be_p, be_r2, n_be = _age_ols(d[d['Task'] == AGE_BE])
        is_coef, is_p, is_r2, n_is = _age_ols(d[d['Task'] == AGE_IS])
        int_coef, int_t, int_p, n_subj, flag = _age_interaction(d)

        rows.append({
            'Region': reg, 'Region_full': reg_full,
            'n_BE': n_be, 'n_IS': n_is, 'n_subj': n_subj,
            'BE_coef': be_coef, 'BE_p': be_p, 'BE_r2': be_r2,
            'IS_coef': is_coef, 'IS_p': is_p, 'IS_r2': is_r2,
            'Int_coef': int_coef, 'Int_t': int_t, 'Int_p': int_p,
            'Int_flag': flag,
        })

    out = pd.DataFrame(rows)

    # FDR across regions, separately for each of the three tests
    for pre in ('BE', 'IS', 'Int'):
        out[f'{pre}_p_FDR'], _ = fdr_nan_safe(out[f'{pre}_p'].to_numpy(),
                                              alpha=age_alpha)

    return out[['Region', 'Region_full', 'n_BE', 'n_IS', 'n_subj',
                'BE_coef', 'BE_p', 'BE_p_FDR', 'BE_r2',
                'IS_coef', 'IS_p', 'IS_p_FDR', 'IS_r2',
                'Int_coef', 'Int_t', 'Int_p', 'Int_p_FDR', 'Int_flag']]


# Run every marker
age_reg_results = {}
for age_marker, age_df in sheets.items():
    age_reg_results[age_marker] = regional_age_table(age_df)

# Console summary: regions with p < alpha (uncorrected / FDR)
age_summary = []
for age_marker, tab in age_reg_results.items():
    row = {'Marker': age_marker}
    for pre in ('BE', 'IS', 'Int'):
        row[f'{pre}_unc'] = int((tab[f'{pre}_p']     < age_alpha).sum())
        row[f'{pre}_FDR'] = int((tab[f'{pre}_p_FDR'] < age_alpha).sum())
    row['flagged_fits'] = int((tab['Int_flag'] != '').sum())
    age_summary.append(row)
age_summary_df = pd.DataFrame(age_summary)

print(f"\n Regional age effects: regions with p < {age_alpha} "
      f"(of {len(reg_name)}) ")
print(age_summary_df.to_string(index=False))

#  Write 
if save_xlsx:
    os.makedirs(lme_dir, exist_ok=True)
    age_out_path = os.path.join(lme_dir, 'BEvIS_regional_age_effects.xlsx')

    age_all = pd.concat(
        [tab.assign(Marker=m) for m, tab in age_reg_results.items()],
        ignore_index=True
    )
    age_all = age_all[['Marker'] + [c for c in age_all.columns if c != 'Marker']]

    with pd.ExcelWriter(age_out_path, engine='openpyxl') as writer:
        age_all.to_excel(writer, sheet_name='All_markers', index=False)
        age_summary_df.to_excel(writer, sheet_name='Summary', index=False)
        for m, tab in age_reg_results.items():
            tab.to_excel(writer, sheet_name=str(m)[:31], index=False)

    print(f"Saved {len(age_reg_results)} marker sheets -> {age_out_path}")

# %% 9.1 - Brain Plots: Regional Age Effects (BE, IS, Task x Age)
"""
  Plots regional age coefficients from section 9 that pass the set threshold.
  Baby Einstein and Inscapes maps share a colour scale for each marker.
"""
import importlib
import functions.brainplots as brainplots
from scipy import stats
importlib.reload(brainplots)

#  CONFIG 
age_bp_markers = list(age_reg_results)   # or a subset, e.g. ['beta']
age_bp_p_col   = 'p_FDR'                 # 'p_FDR' | 'p' (uncorrected)
age_bp_alpha   = 0.05
save_fig       = True

age_bp_dir = os.path.join(lme_dir, 'brainplots')

AGE_MARKER_TEX = {
    # 'slope':           'Slope',
    # 'offset':          'Offset',
    'theta':           r'$\theta$',
    'alpha':           r'$\alpha$',
    'beta':            r'$\beta$',
    'lowgamma':        r'$\gamma_1$',
    'highgamma':       r'$\gamma_2$',
    'alpha_peak_freq': r'Peak $\alpha$ freq',
    'alpha_peak_amp':  r'Peak $\alpha$ amp',
}


def age_masked_coef(tab, pre, p_col=None, alpha=None):
    """
    Coefficient map for one test ('BE', 'IS' or 'Int'), zeroed wherever the
    chosen p-value is not below alpha. Regions with no fit (NaN) are zeroed too.
    """
    p_col = age_bp_p_col if p_col is None else p_col
    alpha = age_bp_alpha if alpha is None else alpha
    coef  = tab[f'{pre}_coef'].to_numpy(dtype=float)
    pvals = tab[f'{pre}_{p_col}'].to_numpy(dtype=float)
    return np.where(pvals < alpha, coef, 0.0)


def _age_brainplot(dat, vmax, cbar_label, fname):
    # Stretch the colour range so this map sits on a scale whose top is `vmax`.

    vmax_scale = vmax / np.max(np.abs(dat))
    fig = brainplots.surface_brain_plotv2(
        make_4d_atlas_nifti(image.load_img(atlasimg), dat),
        FS_DIR, surf='pial', fade=True, cort=(0.9, 0.9, 0.9),
        cmap='coolwarm', symmetric=True,
        cbar_label=cbar_label,
        datmin=-vmax, datmax=vmax, sym_div=10, vmax_scale=vmax_scale
    )
    if save_fig:
        fig.savefig(os.path.join(age_bp_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig

if save_fig:
    os.makedirs(age_bp_dir, exist_ok=True)

for age_marker in age_bp_markers:
    tab = age_reg_results[age_marker]
    tex = AGE_MARKER_TEX.get(str(age_marker).lower(), str(age_marker))

    dat_be  = age_masked_coef(tab, 'BE')
    dat_is  = age_masked_coef(tab, 'IS')
    dat_int = age_masked_coef(tab, 'Int')

    #  BE and IS: shared scale 
    vmax_task = max(np.max(np.abs(dat_be)), np.max(np.abs(dat_is)))
    for pre, dat in (('BE', dat_be), ('IS', dat_is)):
        n_sig = int(np.sum(dat != 0))
        if n_sig == 0:
            print(f"{age_marker:>16} | {pre:<3} | no regions pass, skipped")
            continue
        _age_brainplot(dat, vmax_task,                 # shared BE/IS limit
               fr'{tex} age slope {pre} (per yr)',
               f'{age_marker}_age_{pre}_bp.png')
        print(f"{age_marker:>16} | {pre:<3} | {n_sig} regions plotted")

    #  Task x Age interaction: own scale 
    n_sig = int(np.sum(dat_int != 0))
    if n_sig == 0:
        print(f"{age_marker:>16} | Int | no regions pass, skipped")
    else:
        _age_brainplot(dat_int, np.max(np.abs(dat_int)),
                       fr'{tex} Task $\times$ Age (per yr)',
                       f'{age_marker}_age_Int_bp.png')
        print(f"{age_marker:>16} | Int | {n_sig} regions plotted: "
              f"{', '.join(tab.loc[dat_int != 0, 'Region'])}")

# %% 9.1b - Back view: beta Task x Age interaction (single use)
"""
 Plots a back view of the significant beta Task x Age regions from section 9.
 """
import os.path as op
import mne
import matplotlib as mpl
from nilearn import surface



#  Masked coefficient map (regions with FDR p < 0.05 only) 
tab  = age_reg_results['beta']
dat  = np.where(tab['Int_p_FDR'].to_numpy() < 0.05,
                tab['Int_coef'].to_numpy(), 0.0)
tmax = np.max(np.abs(dat))
if tmax == 0:
    raise ValueError("No beta region passes Int_p_FDR < 0.05, nothing to plot")

#  Volume -> fsaverage surface -> stc (same as surface_brain_plotv2) 
img  = make_4d_atlas_nifti(image.load_img(atlasimg), dat)
lh   = surface.vol_to_surf(img, op.join(FS_DIR, 'fsaverage', 'surf', 'lh.pial'))
rh   = surface.vol_to_surf(img, op.join(FS_DIR, 'fsaverage', 'surf', 'rh.pial'))
stc  = mne.SourceEstimate(np.hstack([lh, rh]),
                          [np.arange(len(lh)), np.arange(len(rh))],
                          tmin=0, tstep=1)

#  Render the back (caudalrostral) view of both hemispheres 
brain = stc.plot(views='caudal', hemi='both',
                 subject='fsaverage', subjects_dir=FS_DIR,
                 surface='pial', cortex=(0.9, 0.9, 0.9), background='white',
                 colorbar=False, time_label=None, time_viewer=False,
                 transparent=True, colormap='coolwarm',
                 clim={'kind': 'value', 'pos_lims': (0, tmax / 2, tmax)})
shot = brain.screenshot()
brain.close()

# crop the white border
nonwhite = (shot != 255).any(-1)
shot = shot[nonwhite.any(1)][:, nonwhite.any(0)]

# Figure: brain + colourbar 
def _cbar_lab(v):
    if v == 0:
        return '0'
    if 0.01 <= abs(v) < 1000:
        return f'{v:.2f}'
    mant, exp = f'{v:.1e}'.split('e')
    return rf'{mant} × 10$^{{{int(exp)}}}$'

fig = plt.figure(figsize=(9, 6))
ax  = fig.add_axes([0.0, 0.05, 0.66, 0.9])
ax.imshow(shot)
ax.axis(False)

cax  = fig.add_axes([0.70, 0.2, 0.04, 0.6])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-tmax, tmax), cmap='coolwarm'),
    cax=cax, orientation='vertical', ticks=[-tmax, tmax]
)
cbar.ax.set_yticklabels([_cbar_lab(-tmax), _cbar_lab(tmax)])
cbar.ax.tick_params(labelsize=40)
cbar.set_label(r'$\beta$ Task $\times$ Age (per yr)', fontsize=40, labelpad=18)

fig.savefig(os.path.join(lme_dir, 'beta_age_Int_back_bp.png'),
              dpi=150, bbox_inches='tight')
plt.show()

# %% 9.2 - Regional Age Scatter (regions with a Task x Age interaction)
"""
 Plots age effects for regions with a significant Task x Age interaction, saved
  with and without a statistics legend.
"""
#  CONFIG 
scatter_marker  = 'beta'
scatter_regions = None     
scatter_p_col   = 'p_FDR'   # p-value shown in the legend: 'p_FDR' | 'p'
scatter_alpha   = 0.05
scatter_ylabel  = None      
legend_fontsize = 30
save_fig        = True

AGE_TASK_COLORS = {'babyeinstein': '#5B8DB8', 'inscapes': '#E07B54'}
AGE_TASK_NAMES  = {'babyeinstein': 'Baby Einstein', 'inscapes': 'Inscapes'}



def _fmt_coef(x):
    """Coefficient as mathtext: plain decimals, or a x 10^n when very small/large."""
    if not np.isfinite(x):
        return 'n/a'
    if x == 0 or 1e-2 <= abs(x) < 1e3:
        return f'${x:.3g}$'
    mant, exp = f'{x:.1e}'.split('e')
    return fr'${mant}\times10^{{{int(exp)}}}$'


def _fmt_p(p, p_col):
    name = r'$p_{FDR}$' if p_col == 'p_FDR' else r'$p$'
    if not np.isfinite(p):
        return f'{name} n/a'
    return f'{name} < 0.001' if p < 0.001 else f'{name} = {p:.3f}'


def age_stats_legend(ax, row, p_col='p_FDR', fontsize=30):
    """Legend to the right of the axes: BE slope, IS slope, Task x Age term."""
    handles = [
        Line2D([0], [0], color=AGE_TASK_COLORS['babyeinstein'], lw=5,
               marker='o', markersize=14,
               label=(f"{AGE_TASK_NAMES['babyeinstein']}: "
                      f"b = {_fmt_coef(row['BE_coef'])}, "
                      f"{_fmt_p(row[f'BE_{p_col}'], p_col)}")),
        Line2D([0], [0], color=AGE_TASK_COLORS['inscapes'], lw=5,
               marker='o', markersize=14,
               label=(f"{AGE_TASK_NAMES['inscapes']}: "
                      f"b = {_fmt_coef(row['IS_coef'])}, "
                      f"{_fmt_p(row[f'IS_{p_col}'], p_col)}")),
        Line2D([0], [0], linestyle='none',
               label=(fr"Task $\times$ Age: "
                      f"b = {_fmt_coef(row['Int_coef'])}, "
                      f"{_fmt_p(row[f'Int_{p_col}'], p_col)}")),
    ]
    return ax.legend(handles=handles, loc='upper left',
                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                     frameon=False, fontsize=fontsize, handlelength=1.6)


#  Pick regions 
scatter_tab = age_reg_results[scatter_marker]
if scatter_regions is None:
    scatter_regions = scatter_tab.loc[
        scatter_tab[f'Int_{scatter_p_col}'] < scatter_alpha, 'Region'
    ].tolist()
print(f"{scatter_marker}: plotting {len(scatter_regions)} region(s): {scatter_regions}")

scatter_df  = sheets[scatter_marker]
_wb_col     = [c for c in scatter_df.columns if c.startswith('WholeBrain_')][0]
_suffix     = _wb_col[len('WholeBrain_'):]
_ylab       = scatter_ylabel or str(scatter_marker).replace('_', ' ').capitalize()

age_scatter_figs = {}
for reg in scatter_regions:
    y_col = f'{reg}_{_suffix}'
    row   = scatter_tab.loc[scatter_tab['Region'] == reg].iloc[0]
    res_reg = fit_interaction(scatter_df, y_col)

    for with_legend in (False, True):
        ax = plot_age_effects_pub(res_reg, scatter_df, y_col, param_lab=_ylab,
                                  task_colors=AGE_TASK_COLORS,
                                  task_labels=AGE_TASK_NAMES)
        if with_legend:
            age_stats_legend(ax, row, p_col=scatter_p_col,
                             fontsize=legend_fontsize)
        fig = ax.figure
        age_scatter_figs[(reg, with_legend)] = fig

        if save_fig:
            os.makedirs(lme_dir, exist_ok=True)
            fname = (f"{scatter_marker}_{reg}_age"
                     f"{'_leg' if with_legend else ''}.png")
            fig.savefig(os.path.join(lme_dir, fname), dpi=150,
                        bbox_inches='tight')
        plt.show()

# %% 10 - Export Marker Data to Excel
"""
 Exports each participant's marker values to Excel, with one sheet per marker.
"""

#  Build sheets 
# Note: theta/alpha/beta use fooof periodic; lowgamma/highgamma use raw PSD
sheets = {
    'Slope':          build_marker_df('slope',     slope_mean,
                                      fooof_params[:, :, -1]),
    'Offset':         build_marker_df('offset',    offset_mean,
                                      fooof_params[:, :,  0]),
    'Theta':          build_marker_df('theta',     np.mean(theta_psd,     axis=1),
                                      theta_psd),
    'Alpha':          build_marker_df('alpha',     np.mean(alpha_psd,     axis=1),
                                      alpha_psd),
    'Beta':           build_marker_df('beta',      np.mean(beta_psd,      axis=1),
                                      beta_psd),
    'LowGamma':       build_marker_df('lowgamma',  np.mean(lowgamma_psd,  axis=1),
                                      lowgamma_psd),
    'HighGamma':      build_marker_df('highgamma', np.mean(highgamma_psd, axis=1),
                                      highgamma_psd),
    'AlphaPeak_Freq': build_marker_df('alpha_peak_freq', wb_peak_freq,
                                      reg_peak_freq),
    'AlphaPeak_Amp':  build_marker_df('alpha_peak_amp',  wb_peak_amp,
                                      reg_peak_amp),
}

#  Write 
out_path = os.path.join(base_dir, 'BEvIS_brainvalues.xlsx')
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    for sheet_name, df in sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Saved {len(sheets)} sheets → {out_path}")
print(f"  Rows per sheet : {num_sub}  (BE + IS)")
print(f"  Cols per sheet : 7 meta + {num_reg} regions")

# %% QC 1 - Per-Subject FOOOF Fit Grid (QC) — BEvIS Study
"""
Plots each participant's FOOOF fits across all regions for quality control.
Region titles show R² and MAE, colour-coded by fit quality.
"""
import math
sns.set(style='white')

def gaussian(f, c, a, w):
    return a * np.exp(-0.5 * ((f - c) / (w / 2)) ** 2)

#  CONFIG 
log         = True
save_fig    = True
close_after = True

task_arr   = np.array(tasks)
task_labels = sorted(np.unique(task_arr))

# Colours per task — extend if more than 2 tasks exist
TASK_COLORS = {
    task_labels[0]: '#E07B39',   # babyeinstein
    task_labels[1]: '#3A7EBF',   # inscapes
}

# R2 thresholds for subplot title colour
R2_GOOD = 0.95
R2_OK   = 0.85

# Grid: 38 regions → 5 rows × 8 cols (spare cells used for legend)
NCOLS = 8
NROWS = math.ceil(num_reg / NCOLS)

#  Frequency axis for group-mean context traces 
freq_axis = freq_bins_low[0:46]
xlims     = (freq_axis[0], freq_axis[-1])

# Task means of the full FOOOF spectrum used as context behind each fit
context_arr  = fooof_spectrum_psd[:, 0:46, :]          # (sub, freq, reg)
task_grp_means = {
    t: np.nanmean(context_arr[task_arr == t], axis=0)   # (freq, reg)
    for t in task_labels
}

#  Output directory 
out_dir_fooof = os.path.join(base_dir, 'psd', 'subject_psd', 'fooof_fit')
if save_fig:
    os.makedirs(out_dir_fooof, exist_ok=True)
    print(f"Saving FOOOF QC figures → {out_dir_fooof}")

#  Main loop: one figure per subject 
for sub_i, subj_label in enumerate(subjects):

    task       = task_arr[sub_i]
    subj_color = TASK_COLORS[task]
    grp_mean   = task_grp_means[task]

    other_tasks = [t for t in task_labels if t != task]

    fig, axes = plt.subplots(NROWS, NCOLS,
                             figsize=(NCOLS * 3.2, NROWS * 2.8),
                             constrained_layout=True)
    axes_flat = axes.flatten()

    r2_all  = []
    err_all = []

    for reg_idx in range(num_reg):
        ax = axes_flat[reg_idx]

        # Pull model object and reconstruct components 
        fm            = fooof_models[sub_i][reg_idx]
        freqs         = fm.freqs
        spectrum      = fm.power_spectrum
        aperiodic_fit = fm._ap_fit
        peak_params   = fm.peak_params_
        periodic_fit  = (sum(gaussian(freqs, *pk) for pk in peak_params)
                         if len(peak_params) else np.zeros_like(freqs))
        full_fit      = aperiodic_fit + periodic_fit

        #  Fit quality metrics 
        r2  = fm.r_squared_
        err = fm.error_
        r2_all.append(r2)
        err_all.append(err)

        # Colour-code title by R²
        if r2 >= R2_GOOD:
            title_color = '#1a7a1a'   # green
        elif r2 >= R2_OK:
            title_color = '#cc7a00'   # orange
        else:
            title_color = '#cc0000'   # red

        #  Context: faint task-group means 
        for ot in other_tasks:
            ax.plot(freq_axis, task_grp_means[ot][:, reg_idx],
                    color=TASK_COLORS[ot], alpha=0.20, linewidth=1.0, zorder=1)
        ax.plot(freq_axis, grp_mean[:, reg_idx],
                color=subj_color, alpha=0.28, linewidth=1.2, zorder=2)

        # FOOOF components 
        ax.plot(freqs, 10**spectrum,
                color='crimson',   linewidth=1.8, zorder=3, label='Original')
        ax.plot(freqs, 10**aperiodic_fit,
                color='slateblue', linewidth=1.6, linestyle='--', zorder=4, label='Aperiodic')
        ax.plot(freqs, 10**full_fit,
                color='black',     linewidth=1.8, zorder=5, label='Full model')

        if log:
            ax.set_yscale('log')

        ax.set_xlim(*xlims)
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title(f'{reg_name[reg_idx]}\n'
                     fr'R²={r2:.3f}  MAE={err:.3f}',
                     fontsize=6.5, pad=2, color=title_color, fontweight='bold')

    # Legend in first unused cell 
    legend_handles = [
        Line2D([0], [0], color='crimson',   linewidth=2,              label='Original spectrum'),
        Line2D([0], [0], color='slateblue', linewidth=2, linestyle='--', label='Aperiodic fit'),
        Line2D([0], [0], color='black',     linewidth=2,              label='Full model'),
        Line2D([0], [0], color=subj_color,  linewidth=2, alpha=0.35,  label=f'{task} mean (this sub)'),
        *[Line2D([0], [0], color=TASK_COLORS[ot], linewidth=2, alpha=0.25, label=f'{ot} mean')
          for ot in other_tasks],
        Line2D([0], [0], color='#1a7a1a', linewidth=0, marker='s', markersize=8, label=f'R²≥{R2_GOOD} good'),
        Line2D([0], [0], color='#cc7a00', linewidth=0, marker='s', markersize=8, label=f'R²≥{R2_OK} ok'),
        Line2D([0], [0], color='#cc0000', linewidth=0, marker='s', markersize=8, label=f'R²<{R2_OK} poor'),
    ]
    for i in range(num_reg, len(axes_flat)):
        axes_flat[i].set_visible(False)
    if num_reg < len(axes_flat):
        leg_ax = axes_flat[num_reg]
        leg_ax.set_visible(True)
        leg_ax.axis('off')
        leg_ax.legend(handles=legend_handles, loc='center', fontsize=8, frameon=False)
    else:
        fig.legend(handles=legend_handles, loc='lower right', fontsize=8, frameon=False)

    # Suptitle: subject + task + mean fit quality 
    mean_r2  = np.nanmean(r2_all)
    mean_err = np.nanmean(err_all)
    n_poor   = int(np.sum(np.array(r2_all) < R2_OK))
    sup_color = '#cc0000' if n_poor > 3 else ('#cc7a00' if n_poor > 0 else '#1a7a1a')

    fig.suptitle(
        f'{subj_label}  ·  {task}      '
        fr'mean R²={mean_r2:.3f}   mean MAE={mean_err:.3f}   '
        f'poor fits (<{R2_OK}): {n_poor}/{num_reg}',
        fontsize=12, fontweight='bold', color=sup_color, y=1.01
    )

    if save_fig:
        fpath = os.path.join(out_dir_fooof, f'{subj_label}_{task}_R2{mean_r2:.2f}.png')
        fig.savefig(fpath, dpi=120, bbox_inches='tight')
        print(f"  [{sub_i + 1:03d}/{num_sub}] {subj_label} ({task})  "
              f"mean R²={mean_r2:.3f}  poor={n_poor}/{num_reg}")

    if close_after:
        plt.close(fig)
    else:
        plt.show()

print("Done — FOOOF QC figures complete.")

