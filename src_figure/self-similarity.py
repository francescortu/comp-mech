import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
sys.path.append(os.path.abspath(os.path.join("../results")))
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to calculate the percentage
def calculate_percentage(row):
    total = row["target_true"] + row["target_false"] + row["other"]
    if total > 0:
        return (row["target_true"] / total) * 100
    else:
        return 0

#import data
df = pd.read_csv("../results/gpt2_evaluate_mechanism_NEW.csv")
# filter the one with a value in the self-similarity column
df = df[df["self-similarity"] == "yes"]

df["percentage"] = df.apply(lambda row: calculate_percentage(row), axis=1)



y_values = df.groupby("model_name")["percentage"].apply(list).tolist()
#x values are the intervals, that are the same for all models, but we need to know how many intervals there are
x_values = df.groupby("model_name")["interval"].apply(list).tolist()[0]


# Data for the line plot
# x_values = [4, 3, 2, 1]  # X-axis points
# y_values = [
#     [3.68, 3.64, 2.64, 6.56],
#     [3.71, 2.83, 3.28, 7.44],
#     [8.53, 7.94, 7.61, 16.66],
#     [6.76, 4.76, 6.48, 11.01],
#     [21.41, 23.54, 33.97, 42.26]
# ]
base_case_values = [4.13, 4.31, 10.20, 7.26, 30.32]



colors = ['b', 'g', 'r', 'm', 'c']
legend_names = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'pythia-6.9b']

# LINEPLOT
plt.figure(figsize=(10, 6))
for i, y in enumerate(y_values):
    plt.plot(x_values, y, marker='o', color=colors[i], label=legend_names[i])
    plt.axhline(y=base_case_values[i], color=colors[i], linestyle='--', alpha=0.5)

# Adding titles, labels, and legend
plt.title('Percentage of Factual Recalling at Each Level of Similarity')
plt.xlabel('X-axis Points')
plt.ylabel('Y-axis Values (%)')
plt.xticks(x_values)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()


## BAR PLOT
n_groups = 5
index = np.arange(n_groups)
bar_width = 0.15

# Plotting the bar plot
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.bar(index + i * bar_width, [y[i] for y in y_values], bar_width, label=f'Level {i + 1}')

# Adding details to the bar plot
plt.xlabel('Models')
plt.ylabel('Percentages (%)')
plt.title('Bar Plot of Factual Recalling')
plt.xticks(index + bar_width, legend_names)
plt.legend()

# Show the plot
plt.show()
