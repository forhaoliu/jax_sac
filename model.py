from functools import partial

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

from jax_utils import extend_and_repeat


def update_target_network(main_params, target_params, tau):
    return jax.tree_multimap(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = "256-256"
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for h in hidden_sizes:
            if self.orthogonal_init:
                x = nn.Dense(
                    h,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = nn.relu(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(
                    1e-2, "fan_in", "uniform"
                ),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        return output


class FullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = "256-256"
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = FullyConnectedNetwork(
            output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init
        )(x)
        return jnp.squeeze(x, -1)


class TanhGaussianPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = "256-256"
    orthogonal_init: bool = False

    def setup(self):
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim,
            arch=self.arch,
            orthogonal_init=self.orthogonal_init,
        )

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = nn.tanh(log_std)
        log_std_min, log_std_max = -10.0, 2.0
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        return action_distribution.log_prob(actions)

    def __call__(self, rng, observations, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = nn.tanh(log_std)
        log_std_min, log_std_max = -10.0, 2.0
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

        return samples, log_prob


class SamplerPolicy(object):
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(self, params, rng, observations, deterministic):
        return self.policy.apply(params, rng, observations, deterministic, repeat=None)

    def __call__(self, rng, observations, deterministic=False, random=False):
        observations = jax.device_put(observations)
        if random:
            # actions = jnp.full_like(actions, jax.random.uniform(rng, (1,), minval=-1, maxval=1.))
            actions = jax.random.uniform(
                rng, (1, self.policy.action_dim), minval=-1, maxval=1.0
            )
        else:
            actions, _ = self.act(
                self.params, rng, observations, deterministic=deterministic
            )
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)
