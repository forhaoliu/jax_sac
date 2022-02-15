import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from pathlib import Path
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import imageio
import numpy as np
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

from jax_utils import init_rng


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.online = False
        config.prefix = ""
        config.project = "BigAPT"
        config.output_dir = Path("/tmp/BigAPT")
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            self.config.project = "{}--{}".format(
                self.config.prefix, self.config.project
            )

        if self.config.output_dir == "":
            self.config.output_dir = Path(tempfile.mkdtemp())
        else:
            self.config.output_dir = self.config.output_dir / self.config.experiment_id
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode="online" if self.config.online else "offline",
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(self.config.output_dir / filename, "wb") as fout:
            pickle.dump(obj, fout)

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


class VideoRecorder:
    def __init__(
        self,
        wandb_logger,
        render_size=256,
        fps=20,
        camera_id=0,
        is_train=False,
    ):
        self.is_train = is_train
        self.save_dir = wandb_logger.config.output_dir / (
            "train_video" if self.is_train else "test_video"
        )
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.camera_id = camera_id
        self.online = wandb_logger.config.online

    def init(self, env):
        self.frames = []
        self.record(env)

    def log_to_wandb(self):
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log(
            {
                "train/video"
                if self.is_train
                else "eval/video": wandb.Video(
                    frames[::skip, :, ::2, ::2], fps=fps, format="gif"
                )
            }
        )

    def save(self, filename):
        if self.online:
            self.log_to_wandb()
        path = self.save_dir / filename
        imageio.mimsave(str(path), self.frames, fps=self.fps)

    def record(self, env):
        if hasattr(env, "physics"):
            frame = env.physics.render(
                height=self.render_size,
                width=self.render_size,
                camera_id=self.camera_id,
            )
        else:
            frame = env.render()
        self.frames.append(frame)


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    init_rng(seed)


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(
                [
                    "{}: {}".format(key, val)
                    for key, val in get_user_flags(flags, flags_def).items()
                ]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            if prefix is not None:
                output["{}.{}".format(prefix, key)] = val
            else:
                output[key] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}
