import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'Arial']

wm_rates = []
nwm_rates = []

filename = 'runs/12_17_16_55_llada_8b_eli5_wm_test/gen_length_400/results.jsonl'

with open(filename, 'r') as f:
    for line in f:
        data = json.loads(line)
        wm_rates.append(data['detection_wm']['satisfaction_rate'])
        nwm_rates.append(data['detection_no_wm']['satisfaction_rate'])
df = pd.DataFrame({
    'Watermarked': wm_rates,
    'Non-Watermarked': nwm_rates
})

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

color_wm = '#27ae60'
color_no_wm = '#e74c3c'
sns.histplot(nwm_rates, bins=30, stat='density', color=color_no_wm, alpha=0.3, label='Non-Watermarked', ax=ax, edgecolor='white')
mu, std = norm.fit(nwm_rates)
xmin, xmax = ax.get_xlim()
x= np.linspace(xmin, xmax, 200)
p = norm.pdf(x, mu, std)
ax.plot(x,p, color=color_no_wm, linewidth=4, label=f'Non-WM Normal Distribution')
sns.histplot(wm_rates, bins=30, stat='density', color=color_wm, alpha=0.3, label='Watermarked', ax=ax, edgecolor='white')
sns.kdeplot(wm_rates, color=color_wm, linewidth=4, label='Watermarked KDE', ax=ax)
sns.despine(ax=ax, top=True, right=True)
ax.spines['left'].set_linewidth(2.0)
ax.spines['bottom'].set_linewidth(2.0)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')
ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
plt.xlabel('Satisfaction Rates', fontsize=22, fontweight='bold', labelpad=15)
plt.ylabel('Density', fontsize=22, fontweight='bold', labelpad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=15, frameon=True, loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig('runs/12_17_16_55_llada_8b_eli5_wm_test/gen_length_400/graph.png')
