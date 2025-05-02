import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt_chart(processes, title="Gantt Chart", save_path="plots/"):
    if not processes:
        print("[plot_gantt.py] No finished processes to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Find number of cores
    cores = set(p.get("core_id", 0) for p in processes)

    colors = plt.cm.tab20.colors  # 20 distinct colors

    for p in processes:
        start = p["start_time"]
        duration = p["finish_time"] - p["start_time"]
        core = p.get("core_id", 0)

        ax.broken_barh([(start, duration)], (core * 10, 9),
                       facecolors=colors[core % len(colors)],
                       edgecolors="black")

        ax.text(start + duration/2, core * 10 + 4.5, 
                f"P{p.get('pid', '')}", 
                ha="center", va="center", fontsize=6, color="black")

    ax.set_yticks([core * 10 + 5 for core in cores])
    ax.set_yticklabels([f"Core {core}" for core in cores])

    ax.set_xlabel("Time")
    ax.set_ylabel("Cores")
    ax.set_title(title)
    ax.grid(True)

    patches = [mpatches.Patch(color=colors[c % len(colors)], label=f'Core {c}') for c in sorted(cores)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        plt.savefig(f"plots/{title.replace(' ', '_')}_gantt.png")
        print(f"[plot_gantt.py] Gantt chart saved to {save_path}")
    else:
        plt.show()
