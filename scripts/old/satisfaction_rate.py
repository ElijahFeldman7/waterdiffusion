import matplotlib as plt
import seaborn as sns
import pandas as pd
import pathlib as Path
import numpy as np
from scipy.stats import norm, gaussian_kde
import json

def create_satisfaction_rate_plot():
    data_file= Path("runs/12_17_16_55_llada_8b_eli5_wm_test/gen_length_400/results.jsonl")

    satisfaction_rates = []

    if data_file.exists():
        with open(data_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    results = json.loads(line)
                    if 'detection_wm' in results and 'satisfaction_rate' in result['detection_wm']:
                        satisfaction_rates.append(result['detection_wm']['satisfaction_rate'])
                    if 'detection_no_wm' in results and 'satisfaction_rate' in results['detection_no_wm']:
                        satisfaction_rates.append(result['detection_no_wm']['satisfaction_rate'])
                except json.JSONDecodeError:
                    continue
    plot_data = [{'gen_length': 400, 'satisfaction_rate': rate} for rate in satisfaction_rates]
    df = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figure_size=(10,4),dpi=300)
    sns.set_style("white")

    if satisfaction_rates: 
        top_bound = np.quantile()
                

    