import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

td3_files = [
    "AntRandom-v5_td3_1_action_1752764120.csv",
    "AntRandom-v5_td3_2_action_1752885920.csv",
    "AntRandom-v5_td3_3_action_1753032245.csv"
]
meow_files = [
    "AntRandom-v5_meow_1_action_250718_124652.csv",
    "AntRandom-v5_meow_2_action_250719_132601.csv",
    "AntRandom-v5_meow_3_action_250719_235548.csv"
]

def aggregate_runs(files, num_points=500):
    all_interpolated = []
    all_steps = []

    for file in files:
        df = pd.read_csv(file)
        df = df.sort_values('step')
        all_steps.append(df['step'].values)
        all_interpolated.append(df['return'].values)

    min_step = max(min(steps[0] for steps in all_steps), 10_000)
    max_step = min(max(steps[-1] for steps in all_steps), 2_490_000)
    common_steps = np.linspace(min_step, max_step, num_points)

    interpolated = []
    for steps, returns in zip(all_steps, all_interpolated):
        f = interp1d(steps, returns, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp = f(common_steps)
        interpolated.append(interp)

    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0)
    return common_steps, mean, std

td3_steps, td3_mean, td3_std = aggregate_runs(td3_files)
meow_steps, meow_mean, meow_std = aggregate_runs(meow_files)

plt.figure(figsize=(10, 6))
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

plt.plot(td3_steps, td3_mean, label='TD3', linewidth=2.5, color='gray')
plt.fill_between(td3_steps, td3_mean - td3_std, td3_mean + td3_std, alpha=0.3, color='gray')

plt.plot(meow_steps, meow_mean, label='Meow', linewidth=2.5, color='limegreen')
plt.fill_between(meow_steps, meow_mean - meow_std, meow_mean + meow_std, alpha=0.3, color='limegreen')

plt.xlabel('Steps')
plt.ylabel('Return', labelpad=10)
plt.title('AntRandom-v5 with Action Perturbation')
plt.legend(loc='upper left', frameon=False)
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
plt.xlim(left=min(td3_steps[0], meow_steps[0]))
plt.xlim(right=2000000)
plt.tight_layout()

plt.savefig("AntRandomAction.pdf", bbox_inches='tight')
plt.savefig("AntRandomAction.svg", transparent=True)
plt.show()