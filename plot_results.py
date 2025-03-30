import matplotlib.pyplot as plt
import os

def plot_comparison(results_dict, metric):
    algorithms = list(results_dict.keys())
    values = [results_dict[algo][metric] for algo in algorithms]

    plt.figure(figsize=(8, 4))
    plt.bar(algorithms, values)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xlabel("Algorithm")
    plt.xticks(rotation=15)
    plt.grid(True, axis='y')
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{metric.replace(' ', '_')}_comparison.png")
    plt.show()