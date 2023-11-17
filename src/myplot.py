# plot the result as a heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_heatmaps(result, result_std, title, interval, save=False, center=0.0):
    # sns.set()
    # sns.set_style("whitegrid", {"axes.grid": False})
    # add also the std heatmap
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("Average logit difference")
    ax[1].set_title("Std logit difference")
    # make the center of the heatmap 0 and white color
    sns.heatmap(
        result.detach().cpu().numpy(),
        annot=True,
        fmt=".1f",
        cbar=False,
        ax=ax[0],
        center=center,
        vmax=interval,
        vmin=-interval,
        cmap="RdBu_r",
    )
    # label the axes
    ax[0].set_xlabel("Head")
    ax[0].set_ylabel("Layer")
    sns.heatmap(
        result_std.detach().cpu().numpy(),
        annot=True,
        fmt=".2f",
        cbar=False,
        ax=ax[1],
        center=0.0,
        cmap="RdBu_r",
    )
    
    
def barplot_head(examples_cp, examples_mem):
    import seaborn as sns
    import numpy as np
    import pandas as pd

    n_layers = 12
    n_heads = 12
    sns.set()
    sns.set_style("whitegrid", {"axes.grid": False})

    mean_cp = examples_cp.mean(dim=-1).detach().cpu().numpy()
    mean_mem = examples_mem.mean(dim=-1).detach().cpu().numpy()

    std_cp = examples_cp.std(dim=-1).detach().cpu().numpy()
    std_mem = examples_mem.std(dim=-1).detach().cpu().numpy()

    flattened_mean_cp = mean_cp.flatten()
    flattened_mean_mem = mean_mem.flatten()

    # sorting  indices
    sorted_indices_cp = np.argsort(flattened_mean_cp)
    sorted_indices_mem = np.argsort(flattened_mean_mem)[::-1]

    #sorting values 
    sorted_values_cp = flattened_mean_cp[sorted_indices_cp]
    sorted_values_mem = flattened_mean_mem[sorted_indices_mem]

    #labels
    labels = [f"L{layer}H{head}" for layer in range(n_layers) for head in range(n_heads)]

    #sorted labels
    sorted_labels_mem = np.array(labels)[sorted_indices_mem]
    sorted_labels_cp = np.array(labels)[sorted_indices_cp]

    # Prepare DataFrames
    df_cp = pd.DataFrame({
        'labels': sorted_labels_cp,
        'values': sorted_values_cp,
        'std': std_cp.flatten()[sorted_indices_cp]
    })

    df_mem = pd.DataFrame({
        'labels': sorted_labels_mem,
        'values': sorted_values_mem,
        'std': std_mem.flatten()[sorted_indices_mem]
    })

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming examples_cp and examples_mem are pre-defined PyTorch tensors
    # The rest of your code for preparing data remains the same...

    # Filter data
    df_cp_positive = df_cp[df_cp['values'] > 0]
    df_cp_negative = df_cp[df_cp['values'] <= 0]
    df_mem_positive = df_mem[df_mem['values'] > 0]
    df_mem_negative = df_mem[df_mem['values'] <= 0]



    # Create subplots
    fig, ax = plt.subplots(4, 1, figsize=(28, 15))
    plt.subplots_adjust(hspace=0.9, wspace=0.4)

    # CP Negative
    sns.barplot(x='labels', y='values', data=df_cp_negative, yerr=df_cp_negative['std'], ax=ax[0])
    ax[0].set_title("CP Negative")
    ax[0].set_xticklabels(df_cp_negative['labels'], rotation=90)


    # MEM Positive
    sns.barplot(x='labels', y='values', data=df_mem_positive, yerr=df_mem_positive['std'], ax=ax[1])
    ax[1].set_title("MEM Positive")
    ax[1].set_xticklabels(df_mem_positive['labels'], rotation=90)

    # CP Positive
    sns.barplot(x='labels', y='values', data=df_cp_positive, yerr=df_cp_positive['std'], ax=ax[2])
    ax[2].set_title("CP Positive")
    ax[2].set_xticklabels(df_cp_positive['labels'], rotation=90)

    # MEM Negative
    sns.barplot(x='labels', y='values', data=df_mem_negative, yerr=df_mem_negative['std'], ax=ax[3])
    ax[3].set_title("MEM Negative")
    ax[3].set_xticklabels(df_mem_negative['labels'], rotation=90)

    plt.show()
    return df_mem, df_cp
