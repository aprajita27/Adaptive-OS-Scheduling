def simulate_fcfs_algorithm(processes):
    queue = sorted(processes.copy(), key=lambda p: p["arrival_time"])
    time = 0
    finished = []

    for p in queue:
        start = max(time, p["arrival_time"])
        p["start_time"] = start
        time = start + p["burst_time"]
        p["finish_time"] = time
        finished.append(p)

    return finished


def simulate_sjf_algorithm(processes):
    queue = sorted(processes.copy(), key=lambda p: (p["arrival_time"], p["burst_time"]))
    time = 0
    finished = []

    while queue:
        available = [p for p in queue if p["arrival_time"] <= time]
        if not available:
            time = queue[0]["arrival_time"]
            continue
        next_proc = min(available, key=lambda p: p["burst_time"])
        queue.remove(next_proc)
        next_proc["start_time"] = time
        time += next_proc["burst_time"]
        next_proc["finish_time"] = time
        finished.append(next_proc)

    return finished


def simulate_priority_preemptive_algorithm(processes):
    queue = sorted(processes.copy(), key=lambda p: p["arrival_time"])
    for p in queue:
        p["remaining_time"] = p["burst_time"]

    time = 0
    finished = []
    current = None
    ready_queue = []

    while queue or ready_queue or current:
        # Add arriving processes to ready_queue
        ready_queue += [p for p in queue if p["arrival_time"] <= time]
        queue = [p for p in queue if p["arrival_time"] > time]

        # Push current back to ready_queue if it's not done
        if current:
            ready_queue.append(current)

        # Pick highest priority
        if ready_queue:
            ready_queue.sort(key=lambda p: p["priority"])
            current = ready_queue.pop(0)
            if current.get("start_time") is None:
                current["start_time"] = time

            current["remaining_time"] -= 1
            time += 1

            if current["remaining_time"] == 0:
                current["finish_time"] = time
                finished.append(current)
                current = None
        else:
            time += 1

    return finished


def simulate_rr_algorithm(processes, quantum=2):
    from collections import deque
    queue = sorted(processes.copy(), key=lambda p: p["arrival_time"])
    for p in queue:
        p["remaining_time"] = p["burst_time"]

    ready = deque()
    time = 0
    idx = 0
    finished = []

    while idx < len(queue) or ready:
        # Load arriving processes
        while idx < len(queue) and queue[idx]["arrival_time"] <= time:
            ready.append(queue[idx])
            idx += 1

        if ready:
            proc = ready.popleft()

            if proc.get("start_time") is None:
                proc["start_time"] = time

            slice_time = min(quantum, proc["remaining_time"])
            proc["remaining_time"] -= slice_time
            time += slice_time

            # Add newly arrived processes during this slice
            while idx < len(queue) and queue[idx]["arrival_time"] <= time:
                ready.append(queue[idx])
                idx += 1

            if proc["remaining_time"] > 0:
                ready.append(proc)
            else:
                proc["finish_time"] = time
                finished.append(proc)
        else:
            time = queue[idx]["arrival_time"]

    return finished
