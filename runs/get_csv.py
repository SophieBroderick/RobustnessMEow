from tbparse import SummaryReader
import seaborn as sns
import matplotlib.pyplot as plt

import os
from tbparse import SummaryReader
import numpy as np

def plot(steps=1000000, yticks=None, xticks=None, palette=None):
    print("plotting...")

    log_dir = os.path.join(".")
    df = SummaryReader(log_dir, pivot=True).scalars

    df = df[["step", "return"]]
    df.to_csv('output.csv', index=False)

    fig = plt.figure(figsize=(5,5.5))
    ax = plt.gca()
    sns.set_theme(style='whitegrid')
    plt.grid(color='lightgray')

    g = sns.lineplot(data=df, x='step', y='return', palette=palette)
    g.set(xlim=(0, steps))
    g.set(ylim=(yticks[0], yticks[-1]))
    if xticks is not None:
        g.set_xticks(xticks)
    if yticks is not None:
        g.set_yticks(yticks)
    plt.legend([],[], frameon=False)

    plt.xlabel('')
    plt.ylabel('')
    plt.close(fig)
    print("Finish plotting.")

def main(steps, yticks, xticks, palette=None):
    plot(steps=steps, yticks=yticks, xticks=xticks, palette=palette)

if __name__ == '__main__':

    steps = 4000000
    yticks = np.arange(-1500, 7500+1500, 1500)
    xticks = np.arange(0, steps+1, 1000000)

    palette = ['xkcd:jade', 'xkcd:deep sky blue', 'xkcd:coral', 'xkcd:orange', 'xkcd:violet', 'xkcd:mauve']
    main(steps, yticks, xticks, palette=palette)