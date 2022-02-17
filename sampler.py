import jax


class RolloutStorage(object):
    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_time_step = None

    def sample_traj(
        self,
        rng,
        policy,
        n_trajs,
        deterministic=False,
        replay_storage=None,
        random=False,
    ):
        r_traj = 0
        time_step, done = self.env.reset(), False
        self._current_time_step = time_step

        if replay_storage is not None:
            replay_storage.add(time_step)

        for _ in range(n_trajs):
            while True:
                self._traj_steps += 1
                rng, split_rng = jax.random.split(rng)
                action = policy(
                    split_rng,
                    self._current_time_step["observation"],
                    deterministic=deterministic,
                    random=random,
                ).reshape(-1)
                time_step = self.env.step(action)
                self._current_time_step = time_step
                r_traj += time_step["reward"]
                done = time_step.last()

                if replay_storage is not None:
                    replay_storage.add(time_step)

                if done or self._traj_steps >= self.max_traj_length:
                    self._traj_steps = 0

                    time_step, done = self.env.reset(), False
                    self._current_time_step = time_step

                    if replay_storage is not None:
                        replay_storage.add(time_step)

                    break

        data = dict(r_traj=r_traj / n_trajs)
        return data, rng

    def sample_step(
        self,
        rng,
        policy,
        n_steps,
        deterministic=False,
        replay_storage=None,
        random=False,
    ):
        for _ in range(n_steps):
            self._traj_steps += 1
            rng, split_rng = jax.random.split(rng)
            action = policy(
                split_rng,
                self._current_time_step["observation"],
                deterministic=deterministic,
                random=random,
            ).reshape(-1)
            time_step = self.env.step(action)
            self._current_time_step = time_step
            done = time_step.last()

            if replay_storage is not None:
                replay_storage.add(time_step)

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0

                time_step, done = self.env.reset(), False
                self._current_time_step = time_step

                if replay_storage is not None:
                    replay_storage.add(time_step)
        data = dict()
        return data, rng

    @property
    def env(self):
        return self._env
