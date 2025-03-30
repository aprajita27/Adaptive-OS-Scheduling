import numpy as np
import gym
from gym import spaces

class CPUSchedulingEnv(gym.Env):
    def __init__(self, all_processes):
        super().__init__()
        self.all_processes = all_processes
        self.num_features = 6  # [arrival_time, burst_time, priority, memory, cpu_req, scheduled]
        self.max_processes = len(all_processes)
        self.max_obs = self.max_processes * self.num_features

        self.context_switch_cost = 2
        self.current_process = None

        print(f"[env.py] Detected {self.max_processes} processes. Setting obs shape = ({self.max_obs},)")

        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.max_obs,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.max_processes)
        self.reset()

    def reset(self):
        self.current_time = 0
        self.finished = []
        self.ready_queue = []
        self.current_process = None
        self.processes = []

        for p in self.all_processes:
            proc = p.copy()
            proc["scheduled"] = False
            proc["start_time"] = None
            proc["finish_time"] = None
            proc["remaining_time"] = proc["burst_time"]
            self.processes.append(proc)

        self.max_arrival = max(p["arrival_time"] for p in self.processes)

        return self._get_obs()

    def _inject_processes(self):
        for p in self.processes:
            if p["arrival_time"] == self.current_time and not p["scheduled"]:
                p["scheduled"] = True
                self.ready_queue.append(p)

    def _get_obs(self):
        obs = []
        for p in self.processes:
            if p["finish_time"] is not None:
                continue  # mask finished processes

            obs += [
                min(p["arrival_time"] / max(1, self.max_arrival), 1.0),
                min(p["remaining_time"] / 100, 1.0) if p["remaining_time"] else 0,
                min(p["priority"] / 100, 1.0),
                min(p["memory"] / 4000, 1.0),
                min(p["cpu_req"] / 100, 1.0),
                int(p.get("scheduled", False)),
            ]

        return np.array(obs[:self.max_obs] + [0] * max(0, self.max_obs - len(obs)), dtype=np.float32)

    def step(self, action):
        self._inject_processes()

        if not self.ready_queue:
            self.current_time += 1
            return self._get_obs(), -2, False, {}

        valid_index = action % len(self.ready_queue)
        selected = self.ready_queue[valid_index]

        reward = -1  # default step penalty

        if self.current_process and self.current_process != selected:
            self.current_time += self.context_switch_cost
            reward -= 0.5 * self.context_switch_cost

        if selected.get("start_time") is None:
            selected["start_time"] = self.current_time

        selected["remaining_time"] -= 1
        self.current_time += 1

        if selected["remaining_time"] == 0:
            selected["finish_time"] = self.current_time
            self.finished.append(selected)
            self.ready_queue.remove(selected)
            self.current_process = None

            turnaround = selected["finish_time"] - selected["arrival_time"]
            waiting = selected["start_time"] - selected["arrival_time"]
            reward += 300 - (turnaround + waiting)  # sharper reward
        else:
            self.current_process = selected

        done = len(self.finished) == self.max_processes
        if self.current_time > 10000:
            return self._get_obs(), -1000, True, {}

        if done:
            return self._get_obs(), 1000, True, {}

        return self._get_obs(), reward, done, {}

    @property
    def finished_processes(self):
        return self.finished