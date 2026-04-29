import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
input_path = "/Users/eli/research/waterdiffusion/runs/subtext_robustness_compare_04_25_04_38/tpr_at_1fpr_by_attack_ratio.csv"
data = []
with open(input_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            "dataset": row["dataset"],
            "detector": row["detector"],
            "tpr": float(row["tpr_at_1pct_fpr"]),
            "attack_ratio": float(row["attack_ratio"])
        })
df = pd.DataFrame(data)
for detector, subdf in df.groupby("detector"):
    subdf = subdf.sort_values("attack_ratio")
    plt.plot(subdf["attack_ratio"], subdf["tpr"], label=detector)
plt.xlabel("attack ratio")
plt.ylabel("tpr@1")
plt.title("robustness genlen = 300, to cropping ")
plt.legend()
plt.savefig("/Users/eli/research/waterdiffusion/runs/subtext_robustness_compare_04_25_04_38/robust_graph.png")

