1
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import norm, gaussian_kde

def create_satisfaction_rate_plot():
    """
    Creates a plot with a stripplot, one-sided violin (KDE), and normal distribution for satisfaction rates.
    """
    data_file = Path("runs/12_12_llada_8b_eli5_wm_test/gen_length_400/results.jsonl")
    
    satisfaction_rates = []
    if data_file.exists():
        with open(data_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    result = json.loads(line)
                    if 'detection_wm' in result and 'satisfaction_rate' in result['detection_wm']:
                        satisfaction_rates.append(result['detection_wm']['satisfaction_rate'])
                except json.JSONDecodeError:
                    print(f"Skipping malformed line: {line.strip()}")

    gen_lengths = [100, 200, 300, 400]
    
    # Create a pandas DataFrame
    plot_data = []
    if satisfaction_rates:
        for rate in satisfaction_rates:
            plot_data.append({"gen_length": 400, "satisfaction_rate": rate})
    
    df = pd.DataFrame(plot_data)
    if not df.empty:
        df['gen_length'] = pd.Categorical(df['gen_length'], categories=gen_lengths, ordered=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Calculate x_pos for gen_length 400
    x_pos = gen_lengths.index(400)

    if satisfaction_rates:
        # Define the y-range for distributions based on quantiles for trimmed tails
        y_min_q = np.quantile(satisfaction_rates, 0.01)
        y_max_q = np.quantile(satisfaction_rates, 0.99)
        y_dist = np.linspace(y_min_q, y_max_q, 100)
        
        # --- Dot plot (stripplot) with transparency ---
        sns.stripplot(x="gen_length", y="satisfaction_rate", data=df, jitter=0.1, size=5, alpha=0.2)

        # --- One-sided Violin Plot (KDE curve) ---
        kde = gaussian_kde(satisfaction_rates)
        p_kde = kde(y_dist)
        p_kde_scaled = p_kde / p_kde.max() * 0.4 # max width of 0.4
        plt.plot(x_pos + p_kde_scaled, y_dist, color='green', lw=2)
        
        # --- Normal Distribution curve ---
        mu, std = norm.fit(satisfaction_rates)
        p_norm = norm.pdf(y_dist, mu, std)
        p_norm_scaled = p_norm / p_norm.max() * 0.4 # max width of 0.4
        plt.plot(x_pos + p_norm_scaled, y_dist, color='red', lw=2)


    plt.xlabel("Generation Length")
    plt.ylabel("Satisfaction Rate")
    plt.title("Watermark Satisfaction Rate by Generation Length")
    plt.xticks(ticks=range(len(gen_lengths)), labels=gen_lengths)
    plt.xlim(-0.5, len(gen_lengths) - 0.5 + 0.4) # Adjusted to account for the bell curve width

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = data_file.parent / "satisfaction_rate_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    create_satisfaction_rate_plot()
