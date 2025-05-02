
import numpy as np
import gym
from gym import spaces

class MultiCoreSchedulingEnv(gym.Env):
    def __init__(self, all_processes, n_cores=4):
        super().__init__()
        self.all_processes = all_processes
        self.n_cores = n_cores
        self.context_switch_cost = 2

        self.num_features = 6
        self.max_processes = len(all_processes)
        self.max_queue_size = self.max_processes // self.n_cores + 5

        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=(self.n_cores, self.max_queue_size * self.num_features),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_queue_size ** self.n_cores)

        self.reset()

    def reset(self):
        self.current_time = 0
        self.finished = []
        self.cores_ready_queues = [[] for _ in range(self.n_cores)]
        self.cores_current_processes = [None] * self.n_cores
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
                core_id = np.argmin([len(q) for q in self.cores_ready_queues])
                self.cores_ready_queues[core_id].append(p)
                p["scheduled"] = True

    def _get_obs(self):
        obs = []
        for queue in self.cores_ready_queues:
            queue_obs = []
            for p in queue[:self.max_queue_size]:
                queue_obs += [
                    min(p["arrival_time"] / max(1, self.max_arrival), 1.0),
                    min(p["remaining_time"] / 100, 1.0),
                    min(p["priority"] / 100, 1.0),
                    min(p["memory"] / 4000, 1.0),
                    min(p["cpu_req"] / 100, 1.0),
                    int(p.get("scheduled", False))
                ]
            queue_obs += [0] * (self.max_queue_size * self.num_features - len(queue_obs))
            obs.append(queue_obs)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self._inject_processes()
        rewards = []

        decoded_actions = []
        flat_action = int(action)
        for _ in range(self.n_cores):
            decoded_actions.append(flat_action % self.max_queue_size)
            flat_action //= self.max_queue_size
        decoded_actions = decoded_actions[::-1]

        for core_id, selected_index in enumerate(decoded_actions):
            queue = self.cores_ready_queues[core_id]
            if not queue:
                rewards.append(-3)
                continue

            selected_index = selected_index % len(queue)
            selected = queue[selected_index]
            reward = -1

            if self.cores_current_processes[core_id] and self.cores_current_processes[core_id] != selected:
                reward -= 0.5 * self.context_switch_cost
                self.current_time += self.context_switch_cost

            if selected.get("start_time") is None:
                selected["start_time"] = self.current_time
                selected["core_id"] = core_id

            selected["remaining_time"] -= 1
            self.current_time += 1

            if selected["remaining_time"] == 0:
                selected["finish_time"] = self.current_time
                self.finished.append(selected)
                self.cores_ready_queues[core_id].remove(selected)
                self.cores_current_processes[core_id] = None
                turnaround = selected["finish_time"] - selected["arrival_time"]
                wait_penalty = (self.current_time - selected["arrival_time"]) * 0.05
                efficiency_bonus = (100 - selected["cpu_req"]) * 0.01
                reward += 300 - turnaround - wait_penalty + efficiency_bonus
            else:
                self.cores_current_processes[core_id] = selected

            rewards.append(reward)

        done = len(self.finished) == self.max_processes
        if self.current_time > 10000:
            return self._get_obs(), np.array([-1000]), True, {}

        if done:
            return self._get_obs(), np.array([1000]), True, {}

        return self._get_obs(), np.array([sum(rewards)]), False, {}

    @property
    def finished_processes(self):
        return self.finished
