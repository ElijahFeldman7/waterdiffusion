import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto Condensed', 'Arial']

def create_satisfaction_rate_plot():
    data_file = Path("runs/12_12_llada_8b_eli5_wm_test/gen_length_400/results.jsonl")
    
    satisfaction_rates = []
    if data_file.exists():
        with open(data_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    result = json.loads(line)
                    if 'detection_wm' in result and 'satisfaction_rate' in result['detection_wm']:
                        satisfaction_rates.append(result['detection_wm']['satisfaction_rate'])
                except json.JSONDecodeError:
                    continue

    gen_lengths = [100, 200, 300, 400]
    plot_data = [{"gen_length": 400, "satisfaction_rate": rate} for rate in satisfaction_rates]
    df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    sns.set_style("white")

    x_pos = gen_lengths.index(400)

    if satisfaction_rates:
        y_min_q = np.quantile(satisfaction_rates, 0.001)
        y_max_q = np.quantile(satisfaction_rates, 0.999)
        y_dist = np.linspace(y_min_q, y_max_q, 100)
        
        sns.stripplot(x="gen_length", y="satisfaction_rate", data=df, 
                      jitter=0.08, size=4, alpha=0.15, color='#34495e', zorder=1)

        kde = gaussian_kde(satisfaction_rates)
        p_kde = kde(y_dist)
        p_kde_scaled = p_kde / p_kde.max() * 0.4
        plt.plot(x_pos + p_kde_scaled, y_dist, color='#27ae60', lw=3, label='Empirical (KDE)', zorder=3)
        
        mu, std = norm.fit(satisfaction_rates)
        p_norm = norm.pdf(y_dist, mu, std)
        p_norm_scaled = p_norm / p_norm.max() * 0.4
        plt.plot(x_pos + p_norm_scaled, y_dist, color='#e74c3c', lw=3, ls='--', label='Normal Fit', zorder=2)

    sns.despine(ax=ax, top=True, right=True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    plt.xlabel("Generation Length (Tokens)", fontsize=22, fontweight='bold', labelpad=15)
    plt.ylabel("Satisfaction Rate ($\hat{p}$)", fontsize=22, fontweight='bold', labelpad=15)
    plt.xticks(ticks=range(len(gen_lengths)), labels=gen_lengths, fontsize=16)
    plt.yticks(fontsize=16)
    
    ax.tick_params(width=2.0, length=8)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    
    plt.xlim(-0.5, len(gen_lengths) - 0.5 + 0.5) 
    plt.legend(fontsize=14, frameon=True, loc='upper left')
    plt.tight_layout()
    
    output_path = data_file.parent / "satisfaction_rate_distribution.png"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Poster plot saved to {output_path}")

if __name__ == "__main__":
    create_satisfaction_rate_plot()