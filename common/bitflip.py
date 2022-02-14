import numpy as np
from dm_control.suite.wrappers import action_scale
from dm_env import StepType, specs

from common.dmc import (
    ActionDTypeWrapper,
    ActionRepeatWrapper,
    ExtendedTimeStep,
    ExtendedTimeStepWrapper,
    ObservationDTypeWrapper,
)


class Env:
    def __init__(self, config, test=False):
        self.size = self.max_steps = config.max_ep_len
        self.test = test
        self.reset()

    def step_type(self):
        if self._step == 0:
            return StepType.FIRST
        elif self._step < self.max_steps:
            return StepType.MID
        else:
            return StepType.LAST

    def reset(self):
        self._step = 0
        state = np.random.randint(2, size=self.size, dtype=np.int32)
        goal = np.random.randint(2, size=self.size, dtype=np.int32)
        self.goal = goal
        self._state = np.copy(state)
        action = np.zeros((1,), dtype=np.int)
        reward = self.check_success(self._state, self.goal)
        step_type = self.step_type()
        if self.test and reward == 1:
            step_type = StepType.LAST
        return ExtendedTimeStep(
            observation=state,
            action=action,
            reward=reward,
            discount=1,
            step_type=step_type,
        )

    def check_success(self, state, goal):
        if np.sum(state == goal) == self.size:
            return 1
        else:
            return 0

    def step(self, action):
        self._step += 1
        state = np.copy(self._state)
        state[action] = 1 - state[action]
        self._state = np.copy(state)
        reward = self.check_success(self._state, self.goal)
        step_type = self.step_type()
        if self.test and reward == 1:
            step_type = StepType.LAST
        return ExtendedTimeStep(
            observation=state,
            action=action,
            reward=reward,
            discount=1,
            step_type=step_type,
        )

    def observation_spec(self):
        return specs.Array(shape=(self.size,), dtype=np.int32, name="observation")

    def action_spec(self):
        return specs.Array(shape=(1,), dtype=np.int32, name="action")


def make_bitflip(config, test=False):
    env = Env(config, test)
    env = ExtendedTimeStepWrapper(env)
    return env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_ep_len", type=int, default=123)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    env = make_bitflip(args)
    print("observation_spec", env.observation_spec())
    print("action_spec", env.action_spec())
    print("observation_shape reset", env.reset()["observation"].shape)
    print(
        "observation_shape step",
        env.step(np.zeros((env.action_spec().shape[0],), dtype=np.int))[
            "observation"
        ].shape,
    )
