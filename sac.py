from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.training.common_utils import shard, shard_prng_key
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from jax_utils import mse_loss, value_and_multi_grad
from model import Scalar, update_target_network


@dataclass
class SAC:
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = "adam"
        config.soft_target_update_rate = 5e-3
        config.nstep = 3

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def update_default_config(self, updates):
        self.config = self.get_default_config(updates)
        HashableConfig = namedtuple(
            "HashableConfig",
            [
                "use_automatic_entropy_tuning",
                "target_entropy",
                "alpha_multiplier",
                "backup_entropy",
                "soft_target_update_rate",
            ],
        )
        self._hashable_config = HashableConfig(
            self.config.use_automatic_entropy_tuning,
            self.config.target_entropy,
            self.config.alpha_multiplier,
            self.config.backup_entropy,
            self.config.soft_target_update_rate,
        )

    def create_state(self, policy, qf, observation_dim, action_dim, rng, obs_type):
        state = {}

        optimizer_class = {
            "adam": optax.adam,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        dummy_obs = jnp.zeros((10, *observation_dim)) if obs_type == "states" else jnp.zeros((10, *observation_dim))
        self._copy_encoder = False if obs_type == "states" else True

        rng, split_rng = jax.random.split(rng)
        policy_params = policy.init(
            split_rng, split_rng, dummy_obs,
        )
        state["policy"] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=policy.apply,
        )

        rng, split_rng = jax.random.split(rng)
        qf_params = qf.init(
            split_rng, dummy_obs, jnp.zeros((10, action_dim))
        )

        state["qf"] = TrainState.create(
            params=qf_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=qf.apply,
        )
        self._target_qf_params = deepcopy({"qf": qf_params})
        self._target_qf_params = jax_utils.replicate(self._target_qf_params)

        model_keys = ["policy", "qf"]

        if self.config.use_automatic_entropy_tuning:
            log_alpha = Scalar(0.0)
            rng, split_rng = jax.random.split(rng)
            state["log_alpha"] = TrainState.create(
                params=log_alpha.init(split_rng),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=log_alpha.apply,
            )
            model_keys.append("log_alpha")

        self._model_keys = tuple(model_keys)

        state = jax_utils.replicate(state)

        return state, rng

    def train(self, state, batch, rng):
        rng = shard_prng_key(rng)
        batch = jax.tree_map(shard, batch)

        state, metrics, rng, self._target_qf_params = train_step(
            state,
            rng,
            batch,
            self._target_qf_params,
            self._hashable_config,
            self._model_keys,
            self._copy_encoder,
        )

        metrics = jax_utils.unreplicate(metrics)
        rng = jax_utils.unreplicate(rng)

        return state, rng, metrics


@partial(jax.pmap, static_broadcasted_argnums=(4, 5, 6), axis_name="batch")
def train_step(state, rng, batch, target_qf_params, train_config, model_keys, copy_encoder):
    def loss_fn(params, rng):
        obs = batch["obs"]
        action = batch["action"]
        reward = jnp.squeeze(batch["reward"], axis=1)
        discount = jnp.squeeze(batch["discount"], axis=1)
        next_obs = batch["next_obs"]

        loss = {}

        rng, split_rng = jax.random.split(rng)
        new_action, log_pi = state["policy"].apply_fn(params["policy"], split_rng, obs)

        if train_config.use_automatic_entropy_tuning:
            alpha_loss = (
                -state["log_alpha"].apply_fn(params["log_alpha"])
                * (log_pi + train_config.target_entropy).mean()
            )
            loss["log_alpha"] = alpha_loss
            alpha = (
                jnp.exp(state["log_alpha"].apply_fn(params["log_alpha"]))
                * train_config.alpha_multiplier
            )
        else:
            alpha_loss = 0.0
            alpha = train_config.alpha_multiplier

        """ Policy loss """
        q_new_action = state["qf"].apply_fn(params["qf"], obs, new_action)
        policy_loss = (alpha * log_pi - q_new_action).mean()

        loss["policy"] = policy_loss

        """ Q function loss """
        q1_pred, q2_pred = state["qf"].apply_fn(params["qf"], obs, action)

        rng, split_rng = jax.random.split(rng)
        new_next_action, next_log_pi = state["policy"].apply_fn(
            params["policy"], split_rng, next_obs
        )
        target_q1, target_q2 = state["qf"].apply_fn(target_qf_params["qf"], next_obs, new_next_action)
        target_q_values = jnp.minimum(target_q1, target_q2)

        if train_config.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        q_target = jax.lax.stop_gradient(reward + discount * target_q_values)
        qf1_loss = mse_loss(q1_pred, q_target)
        qf2_loss = mse_loss(q2_pred, q_target)

        loss["qf"] = qf1_loss + qf2_loss

        return tuple(loss[key] for key in model_keys), locals()

    rng, split_rng = jax.random.split(rng)

    params = {key: state[key].params for key in model_keys}
    (_, aux_values), grads = value_and_multi_grad(
        loss_fn, len(model_keys), has_aux=True
    )(params, split_rng)
    grads = jax.lax.pmean(grads, "batch")

    state = {
        key: state[key].apply_gradients(grads=grads[i][key])
        for i, key in enumerate(model_keys)
    }

    new_target_qf_params = {}
    new_target_qf_params["qf"] = update_target_network(
        state["qf"].params,
        target_qf_params["qf"],
        train_config.soft_target_update_rate,
    )

    if copy_encoder:
        state["policy"].params["params"].copy({"Encoder": state["qf"].params["params"]["Encoder"]})

    metrics = jax.lax.pmean(
        dict(
            log_pi=aux_values["log_pi"].mean(),
            policy_loss=aux_values["policy_loss"],
            qf1_loss=aux_values["qf1_loss"],
            qf2_loss=aux_values["qf2_loss"],
            alpha_loss=aux_values["alpha_loss"],
            alpha=aux_values["alpha"],
            average_qf1=aux_values["q1_pred"].mean(),
            average_qf2=aux_values["q2_pred"].mean(),
            average_target_q=aux_values["target_q_values"].mean(),
        ),
        axis_name="batch",
    )

    return state, metrics, rng, new_target_qf_params
