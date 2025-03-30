from generate_process import generate_dynamic_processes
from traditional_algorithms import (
    simulate_fcfs_algorithm,
    simulate_sjf_algorithm,
    simulate_priority_preemptive_algorithm,
    simulate_rr_algorithm,
)
from train_ppo import train_and_run_ppo
from plot_results import plot_comparison
import copy

def print_metrics(processes, name="Scheduler"):
    valid = [p for p in processes if p.get("start_time") is not None and p.get("finish_time") is not None]
    if not valid:
        print(f"\n{name} Results: No processes were scheduled.")
        return

    n = len(valid)
    total_wait = sum(p["start_time"] - p["arrival_time"] for p in valid)
    total_turnaround = sum(p["finish_time"] - p["arrival_time"] for p in valid)
    total_response = total_wait  # response time = wait time in non-preemptive
    total_burst = sum(p.get("burst_time", 0) for p in valid)

    throughput = n / total_burst if total_burst else 0
    print(f"\n{name} Results:")
    print(f"Throughput = {throughput:.4f}")
    print(f"Average waiting time = {total_wait / n:.4f}")
    print(f"Average turn around time = {total_turnaround / n:.4f}")
    print(f"Average response time = {total_response / n:.4f}")



procs = generate_dynamic_processes(n=100)

for i, proc in enumerate(procs):
    print(f"{i+1}: Arrival={proc.get('arrival_time')}, Burst={proc.get('burst_time')}, "
          f"Priority={proc.get('priority')}, Memory={proc.get('memory')}, "
          f"CPU Req={proc.get('cpu_req')}, Start={proc.get('start_time', '-')}, "
          f"Finish={proc.get('finish_time', '-')}")


fcfs_result = simulate_fcfs_algorithm(copy.deepcopy(procs))
print_metrics(fcfs_result, "FCFS")

sjf_result = simulate_sjf_algorithm(copy.deepcopy(procs))
print_metrics(sjf_result, "SJF")

priority_result = simulate_priority_preemptive_algorithm(copy.deepcopy(procs))
print_metrics(priority_result, "Priority Preemptive")

rr_result = simulate_rr_algorithm(copy.deepcopy(procs), quantum=2)
print_metrics(rr_result, "Round Robin")

ppo_result = train_and_run_ppo(copy.deepcopy(procs), retrain=False)
print_metrics(ppo_result, "PPO")


# plot results
all_results = {}

for name, result in [
    ("FCFS", fcfs_result),
    ("SJF", sjf_result),
    ("Priority Preemptive", priority_result),
    ("Round Robin", rr_result),
    ("PPO", ppo_result)
]:
    valid = [p for p in result if p.get("start_time") is not None and p.get("finish_time") is not None]
    burst_sum = sum([p.get("burst_time", 0) for p in valid])

    if burst_sum == 0 or len(valid) == 0:
        print(f"Warning: Zero burst time sum or no valid processes for {name}. Skipping metric calc.")
        continue

    all_results[name] = {
        "Throughput": len(valid) / burst_sum,
        "Avg Waiting Time": sum([p["start_time"] - p["arrival_time"] for p in valid]) / len(valid),
        "Avg Turnaround Time": sum([p["finish_time"] - p["arrival_time"] for p in valid]) / len(valid),
        "Avg Response Time": sum([p["start_time"] - p["arrival_time"] for p in valid]) / len(valid),
    }


# Plot all metrics
for metric in ["Throughput", "Avg Waiting Time", "Avg Turnaround Time", "Avg Response Time"]:
    plot_comparison(all_results, metric)