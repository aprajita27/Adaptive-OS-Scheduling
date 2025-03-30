def get_available_processes(processes, current_time):
    return [p for p in processes if p["arrival"] <= current_time and not p.get("scheduled", False)]


def mark_scheduled(process):
    process["scheduled"] = True