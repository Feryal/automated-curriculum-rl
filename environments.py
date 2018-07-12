# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environments and environment helper classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import tensorflow as tf


nest = tf.contrib.framework.nest

DEFAULT_ACTION_SET = (
    (0),    # Down
    (1),    # Up
    (2),    # Left
    (3),    # Right
    (4),    # Use
)


class PyProcessCraftLab(object):
  """CraftLab wrapper for PyProcess."""

  def __init__(self, env_sampler, task_name, config, num_action_repeats=1, seed=0,
               runfiles_path=None, level_cache=None):

    del runfiles_path, level_cache, config
    self._num_action_repeats = num_action_repeats
    self._random_state = np.random.RandomState(seed=seed)
    # config = {k: str(v) for k, v in config.iteritems()}
    self._env = None
    self._env_sampler = env_sampler
    self._task_name = task_name
    self._reset()
    self._observation_spec = self._env.obs_specs()

  def _reset(self):
    """Sample a new environment to behave in."""
    self._env = self._env_sampler.sample_environment(self._task_name)
    # print("Reset environment: task: {}: {}".format(
    #    self._env.task_name, self._env.task))
    return self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))

  def _observation(self):
    obs = self._env.observations()
    return self._flatten_obs(obs)

  def _flatten_obs(self, obs):
    return [obs[k] for k in self._observation_spec]

  def initial(self):
    initial_obs = self._reset()
    return self._flatten_obs(initial_obs)

  def step(self, action, task_name):
    reward, done, observation = self._env.step(
        action, num_steps=self._num_action_repeats)
    if done:
      # This will resample an environment according to the new task.
      self._task_name = task_name
      self._reset()
    return reward, done, self._flatten_obs(observation)

  def close(self):
    self._env.close()

  @staticmethod
  def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
    """Returns a nest of `TensorSpec` with the method's output specification."""
    del constructor_kwargs
    # TODO: hardcoded shapes, fix me!
    # add to config
    observation_spec = [
        tf.contrib.framework.TensorSpec([1076, ], tf.float32),
        tf.contrib.framework.TensorSpec([], tf.string),
    ]

    if method_name == 'initial':
      return observation_spec
    elif method_name == 'step':
      return (
          tf.contrib.framework.TensorSpec([], tf.float32),
          tf.contrib.framework.TensorSpec([], tf.bool),
          observation_spec,
      )


StepOutputInfo = collections.namedtuple('StepOutputInfo',
                                        'episode_return episode_step')
StepOutput = collections.namedtuple('StepOutput',
                                    'reward info done observation')


class FlowEnvironment(object):
  """An environment that returns a new state for every modifying method.

  The environment returns a new environment state for every modifying action and
  forces previous actions to be completed first. Similar to `flow` for
  `TensorArray`.
  """

  def __init__(self, env):
    """Initializes the environment.

    Args:
      env: An environment with `initial()` and `step(action)` methods where
        `initial` returns the initial observations and `step` takes an action
        and returns a tuple of (reward, done, observation). `observation`
        should be the observation after the step is taken. If `done` is
        True, the observation should be the first observation in the next
        episode.
    """
    self._env = env

  def initial(self):
    """Returns the initial output and initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. The reward and transition type in the `StepOutput` is the
      reward/transition type that lead to the observation in `StepOutput`.
    """
    with tf.name_scope('flow_environment_initial'):
      initial_reward = tf.constant(0., dtype=tf.float32)
      initial_info = StepOutputInfo(
          tf.constant(0., dtype=tf.float32), tf.constant(0))
      initial_done = tf.constant(True)
      initial_observation = self._env.initial()

      initial_output = StepOutput(
          initial_reward,
          initial_info,
          initial_done,
          initial_observation)

      # Control dependency to make sure the next step can't be taken before the
      # initial output has been read from the environment.
      with tf.control_dependencies(nest.flatten(initial_output)):
        initial_flow = tf.constant(0., dtype=tf.float32)
      initial_state = (initial_flow, initial_info)
      return initial_output, initial_state

  def step(self, action, state, task_name):
    """Takes a step in the environment.

    Args:
      action: An action tensor suitable for the underlying environment.
      state: The environment state from the last step or initial state.

    Returns:
      A tuple of (`StepOutput`, environment state). The environment state should
      be passed in to the next invocation of `step` and should not be used in
      any other way. On episode end (i.e. `done` is True), the returned reward
      should be included in the sum of rewards for the ending episode and not
      part of the next episode.
    """
    with tf.name_scope('flow_environment_step'):
      flow, info = nest.map_structure(tf.convert_to_tensor, state)

      # Make sure the previous step has been executed before running the next
      # step.
      with tf.control_dependencies([flow]):
        reward, done, observation = self._env.step(action, task_name)

      with tf.control_dependencies(nest.flatten(observation)):
        new_flow = tf.add(flow, 1)

      # When done, include the reward in the output info but not in the
      # state for the next step.
      new_info = StepOutputInfo(info.episode_return + reward,
                                info.episode_step + 1)
      new_state = new_flow, nest.map_structure(
          lambda a, b: tf.where(done, a, b),
          StepOutputInfo(tf.constant(0., dtype=tf.float32), tf.constant(0)),
          new_info)

      output = StepOutput(reward, new_info, done, observation)

      return output, new_state
