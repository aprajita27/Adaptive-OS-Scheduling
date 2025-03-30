import random

def generate_dynamic_processes(n=100, max_arrival=50):
    processes = []
    for i in range(n):
        arrival_time = random.randint(0, max_arrival)
        burst_time = random.randint(1, 10)
        proc = {
            "pid": i + 1,
            "arrival_time": arrival_time,
            "burst_time": burst_time,
            "priority": random.randint(1, 100),
            "memory": random.randint(128, 4096),  
            "cpu_req": random.randint(1, 100),    
            "start_time": None,
            "finish_time": None
        }
        processes.append(proc)

    # Sort by arrival time to simulate real-time scheduling
    processes.sort(key=lambda p: p["arrival_time"])
    return processes

