import pickle
import tempfile
import uuid
from copy import copy
from pathlib import Path

import absl.app
import absl.flags
import tqdm
import jax
import numpy as np
import wandb
from dm_env import specs
from flax import jax_utils

from common.dmc import make
from model import FullyConnectedQFunction, SamplerPolicy, TanhGaussianPolicy
from replay_buffer import ReplayBufferStorage, make_replay_loader
from sac import SAC
from sampler import RolloutStorage
from utils import (
    Timer,
    define_flags_with_default,
    get_user_flags,
    prefix_metrics,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    env="walker_stand",
    obs_type="states",
    max_traj_length=1000,
    replay_buffer_size=1000000,
    seed=42,
    save_model=False,
    policy_arch="256-256",
    qf_arch="256-256",
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,
    n_epochs=2000001,
    n_train_step_per_epoch=1,
    n_sample_step_per_epoch=1,
    eval_period=10000,
    eval_n_trajs=5,
    frame_stack=1,
    action_repeat=1,
    batch_size=256,
    save_replay_buffer=False,
    n_worker=4,
    sac=SAC.get_default_config(),
    online=False,
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    replay_dir = Path(tempfile.mkdtemp())
    log_dir = Path(tempfile.mkdtemp())
    wandb.init(
        config=copy(variant),
        project="SAC",
        dir=log_dir,
        id=uuid.uuid4().hex,
        mode="online" if FLAGS.online else "offline",
    )

    set_random_seed(FLAGS.seed)

    train_env = make(
        FLAGS.env, FLAGS.obs_type, FLAGS.frame_stack, FLAGS.action_repeat, FLAGS.seed
    )
    test_env = make(
        FLAGS.env,
        FLAGS.obs_type,
        FLAGS.frame_stack,
        FLAGS.action_repeat,
        FLAGS.seed + 1000,
    )

    train_sampler = RolloutStorage(train_env, FLAGS.max_traj_length)
    eval_sampler = RolloutStorage(test_env, FLAGS.max_traj_length)
    data_specs = (
        train_env.observation_spec(),
        train_env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
    )
    replay_storage = ReplayBufferStorage(data_specs, replay_dir / "replay")
    replay_loader = make_replay_loader(
        replay_storage,
        FLAGS.replay_buffer_size,
        FLAGS.batch_size * jax.local_device_count(),
        FLAGS.n_worker,
        FLAGS.save_replay_buffer,
        FLAGS.sac.nstep,
        FLAGS.sac.discount,
    )
    replay_iter = None

    def get_replay_iter(replay_iter):
        if replay_iter is None:
            replay_iter = iter(replay_loader)
        return replay_iter

    dummy_env = make(
        FLAGS.env, FLAGS.obs_type, FLAGS.frame_stack, FLAGS.action_repeat, FLAGS.seed
    )
    action_dim = dummy_env.action_spec().shape[0]
    observation_dim = dummy_env.observation_spec().shape[0]

    policy = TanhGaussianPolicy(
        observation_dim,
        action_dim,
        FLAGS.policy_arch,
        FLAGS.orthogonal_init,
    )
    qf = FullyConnectedQFunction(
        observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init
    )

    if FLAGS.sac.target_entropy >= 0.0:
        FLAGS.sac.target_entropy = -np.prod(eval_sampler.env.action_spec().shape).item()

    sac = SAC()
    sac.update_default_config(FLAGS.sac)
    rng = jax.random.PRNGKey(FLAGS.seed)
    state, rng = sac.create_state(policy, qf, observation_dim, action_dim, rng)
    sampler_policy = SamplerPolicy(policy, state["policy"].params)

    # wait until collecting n_worker trajectories for data loader
    _, rng = train_sampler.sample_traj(
        rng,
        sampler_policy.update_params(jax_utils.unreplicate(state)["policy"].params),
        FLAGS.n_worker,
        deterministic=False,
        replay_storage=replay_storage,
        random=True,
    )

    for epoch in tqdm.tqdm(range(FLAGS.n_epochs)):
        metrics = {}
        with Timer() as rollout_timer:
            for _ in range(FLAGS.n_sample_step_per_epoch):
                _, rng = train_sampler.sample_step(
                    rng,
                    sampler_policy.update_params(
                        jax_utils.unreplicate(state)["policy"].params
                    ),
                    1,
                    deterministic=False,
                    replay_storage=replay_storage,
                )
            metrics["env_steps"] = len(replay_storage)
            metrics["epoch"] = epoch

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                replay_iter = get_replay_iter(replay_iter)
                batch = next(replay_iter)
                state, rng, train_metrics = sac.train(state, batch, rng)
                metrics.update(prefix_metrics(train_metrics, "sac"))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                data, rng = eval_sampler.sample_traj(
                    rng,
                    sampler_policy.update_params(
                        jax_utils.unreplicate(state)["policy"].params
                    ),
                    FLAGS.eval_n_trajs,
                    deterministic=True,
                )
                metrics["average_return"] = data["r_traj"]

                if FLAGS.save_model:
                    save_data = {"sac": sac, "variant": variant, "epoch": epoch}
                    with open(log_dir / f"model_epoch_{epoch}.pkl", "wb") as fout:
                        pickle.dump(save_data, fout)

        if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
            metrics["rollout_time"] = rollout_timer()
            metrics["train_time"] = train_timer()
            metrics["eval_time"] = eval_timer()
            metrics["epoch_time"] = rollout_timer() + train_timer() + eval_timer()
            wandb.log(metrics)

    wandb.finish()
    if FLAGS.save_model:
        save_data = {"sac": sac, "variant": variant, "epoch": epoch}
        with open(log_dir / f"model_epoch_{epoch}.pkl", "wb") as fout:
            pickle.dump(save_data, fout)


if __name__ == "__main__":
    absl.app.run(main)
