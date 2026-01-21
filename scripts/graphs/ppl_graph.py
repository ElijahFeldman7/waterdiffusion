import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

file_path = 'runs/12_27_18_12_llada_8b_eli5_wm_multidelta/results.jsonl'

lines_by_delta_wm = defaultdict(list)
lines_by_delta_nwm = defaultdict(list)

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

with open(file_path, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        delta = data['delta']
        
        if 'perplexity_wm' in data:
            lines_by_delta_wm[delta].append(data['perplexity_wm'])
        if 'perplexity_no_wm' in data:
            lines_by_delta_nwm[delta].append(data['perplexity_no_wm'])

sorted_deltas = sorted(lines_by_delta_wm.keys())
wm_medians = [np.median(lines_by_delta_wm[d]) for d in sorted_deltas]
nwm_medians = [np.median(lines_by_delta_nwm[d]) for d in sorted_deltas]

plt.rcParams.update({
    'font.size': 20,
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
})

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

ax.plot(sorted_deltas, wm_medians, label='Watermarked', 
        color='#D41159', lw=5, marker='o', ms=14, zorder=5)

ax.plot(sorted_deltas, nwm_medians, label='No Watermark', 
        color='#1A85FF', lw=5, ls='--', marker='s', ms=14, zorder=4)

ax.set_xlabel(r'Delta ($\delta$)', fontsize=24, labelpad=15)
ax.set_ylabel('Median Perplexity', fontsize=24, labelpad=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, which='major', linestyle='-', alpha=0.15, color='gray')

ax.legend(loc='best', frameon=True, facecolor='white', framealpha=1, edgecolor='#CCCCCC')

ymin = min(wm_medians + nwm_medians) * 0.95
ymax = max(wm_medians + nwm_medians) * 1.05
ax.set_ylim(4, ymax)

plt.tight_layout()
save_path = 'graph.png'
plt.savefig(save_path)
print(f"Median graph saved to: {save_path}")