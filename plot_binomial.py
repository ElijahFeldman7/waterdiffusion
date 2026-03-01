import numpy as np
import matplotlib.pyplot as plt
from math import comb

# Parameters for the binomial distribution B(n, p)
n = 7  # Number of trials
p = 0.5 # Probability of success

# Possible number of successes (k)
k_values = np.arange(0, n + 1)

# Calculate the Probability Mass Function (PMF)
probabilities = [comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in k_values]

# Generate colors for bars
colors = ['skyblue' if k < 5 else '#55c35e' for k in k_values]

# Create the plot
plt.figure(figsize=(7, 6))
plt.bar(k_values, probabilities, color=colors, width=1.0, edgecolor='gray', linewidth=0.8)

# Add labels
plt.xlabel('Number of good pairs(k)')
plt.ylabel('Probability P(X=k)')
plt.xticks(k_values) # Ensure x-axis ticks are integers

# Save the plot
plt.savefig('binomial_distribution.png')
