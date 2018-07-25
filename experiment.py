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

"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import functools
import os
import sys

from craft_env import env_factory
import environments
import numpy as np
import py_process
import sonnet as snt
import tensorflow as tf
import vtrace

import curses

try:
  import dynamic_batching
except tf.errors.NotFoundError:
  tf.logging.warning('Running without dynamic batching.')

from six.moves import range


nest = tf.contrib.framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('logdir', '../results', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Training.
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 1, 'Number of action repeats.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')

# Environment settings.
flags.DEFINE_string(
    'recipes_path', 'craft_env/resources/recipes.yaml',
    'Path to recipes for craft environment')
flags.DEFINE_string(
    'hints_path', 'craft_env/resources/hints.yaml',
    'Path to hints for craft environment')
flags.DEFINE_integer(
    'max_steps', 100,
    'Maximum number of steps before the environment terminates on a failure.')
flags.DEFINE_bool(
    'reuse_environments', False,
    'If set, will use a single environment per task, simplifying tasks'
    'dramatically.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')

# Teacher params.
flags.DEFINE_float(
    'gamma', 0.2, 'Controls the minimum sampling probability for each task')
flags.DEFINE_float('eta', 0.3, 'Learning rate of teacher')
flags.DEFINE_enum(
    'progress_signal', 'advantage', ['reward', 'gradient_norm', 'advantage'],
    'Type of signal to use when tracking down progress of students. ')
flags.DEFINE_integer(
    'switch_tasks_every_k_frames', int(1e4),
    'We will trigger a refresh of the tasks after K environment frames.'
)
flags.DEFINE_integer('save_every_k_teacher_updates', int(50),
                     'Write the Teacher signals to files at this frequency.')


# Flags used for testing/evaluation
flags.DEFINE_integer('test_num_episodes', 30, 'Number of episodes per level.')
flags.DEFINE_integer(
    'evaluate_every_k_frames', int(1e5),
    'Perform a full evaluation on all tasks after K environment frames.'
)


# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'task_name agent_state env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action policy_logits baseline')


def is_single_machine():
  return FLAGS.task == -1


class Agent(snt.RNNCore):
  """Agent with ResNet."""

  def __init__(self, num_actions, obs_specs):
    super(Agent, self).__init__(name='agent')

    self._num_actions = num_actions
    self._obs_specs = obs_specs

    with self._enter_variable_scope():
      # read the model params this from config
      self._core = tf.contrib.rnn.LSTMBlockCell(256)

  def initial_state(self, batch_size):
    return self._core.zero_state(batch_size, tf.float32)

  def _instruction(self, instruction):
    # Split string.
    splitted = tf.string_split(instruction)
    dense = tf.sparse_tensor_to_dense(splitted, default_value='')
    length = tf.reduce_sum(tf.to_int32(tf.not_equal(dense, '')), axis=1)

    # To int64 hash buckets. Small risk of having collisions. Alternatively, a
    # vocabulary can be used.
    num_hash_buckets = 1000
    buckets = tf.string_to_hash_bucket_fast(dense, num_hash_buckets)

    # Embed the instruction. Embedding size 20 seems to be enough.
    # I can embed the task name
    embedding_size = 20
    embedding = snt.Embed(num_hash_buckets, embedding_size)(buckets)

    # Pad to make sure there is at least one output.
    padding = tf.to_int32(tf.equal(tf.shape(embedding)[1], 0))
    embedding = tf.pad(embedding, [[0, 0], [0, padding], [0, 0]])

    core = tf.contrib.rnn.LSTMBlockCell(64, name='task_lstm')
    output, _ = tf.nn.dynamic_rnn(core, embedding, length, dtype=tf.float32)

    # Return last output.
    return tf.reverse_sequence(output, length, seq_axis=1)[:, 0]

  def _torso(self, input_):
    last_action, env_output = input_
    reward, _, _, observations = env_output

    # Repack observations
    observations_dict = {
        obs_name: observations[obs_i]
        for obs_i, obs_name in enumerate(self._obs_specs.keys())
    }
    features_out = snt.Linear(256)(observations_dict['features'])
    features_out = tf.nn.relu(features_out)
    features_out = snt.BatchFlatten()(features_out)

    features_out = snt.Linear(256)(features_out)
    features_out = tf.nn.relu(features_out)

    instruction_out = self._instruction(observations_dict['task_name'])

    # Append clipped last reward and one hot last action.
    clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
    one_hot_last_action = tf.one_hot(last_action, self._num_actions)
    return tf.concat(
        [features_out, clipped_reward, one_hot_last_action, instruction_out],
        axis=1)

  def _head(self, core_output):
    policy_logits = snt.Linear(self._num_actions, name='policy_logits')(
        core_output)
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(core_output), axis=-1)

    # Sample an action from the policy.
    new_action = tf.multinomial(policy_logits, num_samples=1,
                                output_dtype=tf.int32)
    new_action = tf.squeeze(new_action, 1, name='new_action')

    return AgentOutput(new_action, policy_logits, baseline)

  def _build(self, input_, core_state):
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs, core_state = self.unroll(actions, env_outputs, core_state)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

  @snt.reuse_variables
  def unroll(self, actions, env_outputs, core_state):
    _, _, done, _ = env_outputs

    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))

    # Note, in this implementation we can't use CuDNN RNN to speed things up due
    # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
    # changed to implement snt.LSTMCell).
    initial_core_state = self._core.zero_state(
        tf.shape(actions)[1], tf.float32)
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_state = nest.map_structure(functools.partial(tf.where, d),
                                      initial_core_state, core_state)
      core_output, core_state = self._core(input_, core_state)
      core_output_list.append(core_output)

    return snt.BatchApply(self._head)(tf.stack(core_output_list)), core_state


class Teacher(object):

  """Teacher using Exponential-weight algorithm for Exploration and Exploitation (Exp3) algorithm.
  """

  def __init__(self, tasks, gamma=0.3):
    self._tasks = tasks
    self._n_tasks = len(self._tasks)
    self._gamma = gamma
    self._log_weights = np.zeros(self._n_tasks)

  @property
  def task_probabilities(self):
    weights = np.exp(self._log_weights - np.sum(self._log_weights))
    return (1 - self._gamma)*weights / np.sum(weights) + self._gamma/self._n_tasks

  def get_task(self):
    """Samples a task, according to current Exp3 belief.
    """
    task_i = np.random.choice(self._n_tasks, p=self.task_probabilities)
    return self._tasks[task_i]

  def update(self, task, reward):
    task_i = self._tasks.index(task)

    reward_corrected = reward/self.task_probabilities[task_i]
    self._log_weights[task_i] += self._gamma*reward_corrected/self._n_tasks


def build_actor(agent, env, task_name_op, action_set):
  """Builds the actor loop."""
  if isinstance(task_name_op, str):
    task_name_op = tf.constant(task_name_op)
  # Initial values.
  initial_env_output, initial_env_state = env.initial(task_name_op)
  initial_agent_state = agent.initial_state(1)
  initial_action = tf.zeros([1], dtype=tf.int32)
  dummy_agent_output, _ = agent(
      (initial_action,
       nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)),
      initial_agent_state)
  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

  # All state that needs to persist across training iterations. This includes
  # the last environment output, agent state and last agent output. These
  # variables should never go on the parameter servers.
  def create_state(t):
    # Creates a unique variable scope to ensure the variable name is unique.
    with tf.variable_scope(None, default_name='state'):
      return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  persistent_state = nest.map_structure(
      create_state, (initial_env_state, initial_env_output, initial_agent_state,
                     initial_agent_output))

  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_state, agent_output = input_

    # Run agent.
    action = agent_output[0]
    batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            env_output)
    agent_output, agent_state = agent(
        (action, batched_env_output), agent_state)

    # Convert action index to the native action.
    action = agent_output[0][0]
    raw_action = tf.gather(action_set, action)

    env_output, env_state = env.step(raw_action, env_state, task_name_op)

    return env_state, env_output, agent_state, agent_output

  # Run the unroll. `read_value()` is needed to make sure later usage will
  # return the first values and not a new snapshot of the variables.
  first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
  _, first_env_output, first_agent_state, first_agent_output = first_values

  # TODO Useful for debugging I think, single agent step
  # output = step(first_values, 0)
  # _, env_outputs, _, agent_outputs = output

  # Use scan to apply `step` multiple times, therefore unrolling the agent
  # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
  # the output of each call of `step` as input of the subsequent call of `step`.
  # The unroll sequence is initialized with the agent and environment states
  # and outputs as stored at the end of the previous unroll.
  # `output` stores lists of all states and outputs stacked along the entire
  # unroll. Note that the initial states and outputs (fed through `initializer`)
  # are not in `output` and will need to be added manually later.
  output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
  _, env_outputs, _, agent_outputs = output

  # Update persistent state with the last output from the loop.
  assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                  persistent_state, output)

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
    first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
    agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)
    # task_name = nest.map_structure(lambda t: t[0], task_name)

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output),
        (agent_outputs, env_outputs))

    actor_output = ActorOutput(
        task_name=task_name_op, agent_state=first_agent_state,
        env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)

    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, actor_output)


def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))


def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  advantages = tf.stop_gradient(advantages)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)


def build_learner(agent, agent_state, env_outputs, agent_outputs,
                  teacher_task_ph):
  """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

  Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
  """
  learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs,
                                    agent_state)
  teacher_selected_task = tf.identity(teacher_task_ph)

  # Use last baseline value (from the value function) to bootstrap.
  bootstrap_value = learner_outputs.baseline[-1]

  # At this point, the environment outputs at time step `t` are the inputs that
  # lead to the learner_outputs at time step `t`. After the following shifting,
  # the actions in agent_outputs and learner_outputs at time step `t` is what
  # leads to the environment outputs at time step `t`.
  agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
  rewards, infos, done, _ = nest.map_structure(
      lambda t: t[1:], env_outputs)
  learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

  if FLAGS.reward_clipping == 'abs_one':
    clipped_rewards = tf.clip_by_value(rewards, -1, 1)
  elif FLAGS.reward_clipping == 'soft_asymmetric':
    squeezed = tf.tanh(rewards / 5.0)
    # Negative rewards are given less weight than positive rewards.
    # we don't have negative rewards so this is redundant
    clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.

  discounts = tf.to_float(~done) * FLAGS.discounting

  # Compute V-trace returns and weights.
  # Note, this is put on the CPU because it's faster than on GPU. It can be
  # improved further with XLA-compilation or with a custom TensorFlow operation.
  with tf.device('/cpu'):
    vtrace_returns = vtrace.from_logits(
        behaviour_policy_logits=agent_outputs.policy_logits,
        target_policy_logits=learner_outputs.policy_logits,
        actions=agent_outputs.action,
        discounts=discounts,
        rewards=clipped_rewards,
        values=learner_outputs.baseline,
        bootstrap_value=bootstrap_value)

  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  total_loss = compute_policy_gradient_loss(
      learner_outputs.policy_logits, agent_outputs.action,
      vtrace_returns.pg_advantages)
  total_loss += FLAGS.baseline_cost * compute_baseline_loss(
      vtrace_returns.vs - learner_outputs.baseline)
  total_loss += FLAGS.entropy_cost * compute_entropy_loss(
      learner_outputs.policy_logits)

  # Optimization
  num_env_frames = tf.train.get_global_step()
  learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                            FLAGS.total_environment_frames, 0)
  optimizer = tf.train.RMSPropOptimizer(learning_rate, FLAGS.decay,
                                        FLAGS.momentum, FLAGS.epsilon)
  train_op = optimizer.minimize(total_loss)

  # Compute progress signal
  if FLAGS.progress_signal == 'reward':
    # Keep returns at end of episodes.
    # Discard parts of the minibatch using other tasks than what the Teacher
    # expects.
    episode_returns_correct_task = tf.boolean_mask(
        rewards,
        tf.logical_and(done, tf.equal(infos.task_name, teacher_selected_task)))
    progress_signal = tf.where(
        tf.size(episode_returns_correct_task) > 0,
        x=tf.reduce_mean(episode_returns_correct_task, name='progress_reward'),
        y=0)
  elif FLAGS.progress_signal == 'advantage':
    # For Advantage, we will compute returns[t] - returns[t-k] below, when
    # preparing to update the Teacher.
    # So just return reward[t] (again handling the wrong tasks parts)
    episode_returns_correct_task = tf.boolean_mask(
        rewards,
        tf.logical_and(done, tf.equal(infos.task_name, teacher_selected_task)))
    progress_signal = tf.where(
        tf.size(episode_returns_correct_task) > 0,
        x=tf.reduce_mean(episode_returns_correct_task, name='progress_reward'),
        y=0)
  elif FLAGS.progress_signal == 'gradient_norm':
    # compute norm of gradients as the progress signal
    params = tf.trainable_variables()
    gradients = tf.gradients(total_loss, params)
    gradient_norm = tf.global_norm(gradients)
    # TODO renormalize gradients hack, should be done adaptively...
    progress_signal = tf.divide(
        gradient_norm, 500., name='progress_gradient_norm')

  # Merge updating the network and environment frames into a single tensor.
  with tf.control_dependencies([train_op]):
    num_env_frames_and_train = num_env_frames.assign_add(
        FLAGS.batch_size * FLAGS.unroll_length * FLAGS.num_action_repeats)

  # Adding a few summaries.
  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.histogram('action', agent_outputs.action)
  tf.summary.scalar('progress_signal', progress_signal)

  return done, infos, num_env_frames_and_train, progress_signal


def create_environment(env_sampler,
                       initial_task_name=None,
                       seed=0):
  """Creates an environment wrapped in a `FlowEnvironment`."""
  # Sample a task if not provided
  if initial_task_name is None:
    initial_task_name = np.random.choice(env_sampler.task_names)
  # config is empty dict for now
  config = {}
  p = py_process.PyProcess(environments.PyProcessCraftLab, env_sampler,
                           initial_task_name, config, FLAGS.num_action_repeats,
                           seed)

  flow_env = environments.FlowEnvironment(p.proxy)

  # TODO clean me up, useful for debugging
  # obs_reset = p.proxy.initial()
  # rew, done, obs_step = p.proxy.step(0)
  # output_initial, state_initial = flow_env.initial()
  # output_step, state_step = flow_env.step(0, state_initial)

  return flow_env


def update_all_actors_tasks(new_tasks_assignments,
                            actor_task_name_params,
                            session,
                            single_task=False):
  feed_dict = {}
  for actor_i in range(FLAGS.num_actors):
    if single_task:
      actor_task_name_params['task_name'][actor_i] = new_tasks_assignments[0]
    else:
      actor_task_name_params['task_name'][actor_i] = new_tasks_assignments[
          actor_i]
    feed_dict[actor_task_name_params['ph'][actor_i]] = actor_task_name_params[
        'task_name'][actor_i]
  # Update tasks for all actors
  session.run(actor_task_name_params['update'], feed_dict=feed_dict)


@contextlib.contextmanager
def pin_global_variables(device):
  """Pins global variables to the specified device."""
  def getter(getter, *args, **kwargs):
    var_collections = kwargs.get('collections', None)
    if var_collections is None:
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
      with tf.device(device):
        return getter(*args, **kwargs)
    else:
      return getter(*args, **kwargs)

  with tf.variable_scope('', custom_getter=getter) as vs:
    yield vs


def train(action_set):
  """Train."""

  if is_single_machine():
    local_job_device = ''
    shared_job_device = ''

    def is_actor_fn(i): return True
    is_learner = True
    global_variable_device = '/gpu'
    server = tf.train.Server.create_local_server()
    filters = []
  else:
    local_job_device = '/job:%s/task:%d' % (FLAGS.job_name, FLAGS.task)
    shared_job_device = '/job:learner/task:0'

    def is_actor_fn(i): return FLAGS.job_name == 'actor' and i == FLAGS.task
    is_learner = FLAGS.job_name == 'learner'

    # Placing the variable on CPU, makes it cheaper to send it to all the
    # actors. Continual copying the variables from the GPU is slow.
    global_variable_device = shared_job_device + '/cpu'
    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
        'learner': ['localhost:8000']
    })
    server = tf.train.Server(cluster, job_name=FLAGS.job_name,
                             task_index=FLAGS.task)
    filters = [shared_job_device, local_job_device]

  # Only used to find the actor output structure.
  with tf.Graph().as_default():
    # here the meta learning algorithm should propose the task

    env_sampler = env_factory.EnvironmentFactory(
        FLAGS.recipes_path, FLAGS.hints_path, max_steps=FLAGS.max_steps,
        reuse_environments=FLAGS.reuse_environments, seed=1)
    dummy_env = env_sampler.sample_environment()
    obs_spec = dummy_env.obs_specs()
    env = create_environment(env_sampler, seed=1)

    teacher = Teacher(env_sampler.task_names, gamma=FLAGS.gamma)

    agent = Agent(len(action_set), obs_spec)
    structure = build_actor(agent, env, '', action_set)
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]
    shapes = [t.shape.as_list() for t in flattened_structure]

  with tf.Graph().as_default(), \
          tf.device(local_job_device + '/cpu'), \
          pin_global_variables(global_variable_device):
    tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

    # Create Queue and Agent on the learner.
    with tf.device(shared_job_device):
      queue = tf.FIFOQueue(1, dtypes, shapes, shared_name='buffer')
      agent = Agent(len(action_set), obs_spec)

      # Setup the task names variables and assignment logic
      teacher_task_ph = tf.placeholder(
            dtype=tf.string, shape=(), name='teacher_task_name')
      task_names = env_sampler.task_names
      actor_task_name_params = collections.defaultdict(list)
      for actor_i in range(FLAGS.num_actors):
        # Assign initial task name by round-robin
        # initial_task_name = task_names[actor_i % len(task_names)]
        initial_task_name = task_names[0]
        # Setup variables and assignment logic
        actor_task_name_var = tf.get_variable(
            "task_name_actor_{}".format(actor_i),
            shape=(),
            dtype=tf.string,
            initializer=tf.constant_initializer(
                initial_task_name, dtype=tf.string),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES]
        )
        actor_task_name_ph = tf.placeholder(
            dtype=tf.string, shape=(), name='actor_{}_new_task_name'.format(actor_i))
        assign_actor_task_name = tf.assign(
            actor_task_name_var, actor_task_name_ph,
            name='update_task_name_actor_{}'.format(actor_i))
        actor_task_name_params['task_name'].append(initial_task_name)
        actor_task_name_params['var'].append(actor_task_name_var)
        actor_task_name_params['ph'].append(actor_task_name_ph)
        actor_task_name_params['update'].append(assign_actor_task_name)

      if is_single_machine() and 'dynamic_batching' in sys.modules:
        # For single machine training, we use dynamic batching for improved GPU
        # utilization. The semantics of single machine training are slightly
        # different from the distributed setting because within a single unroll
        # of an environment, the actions may be computed using different weights
        # if an update happens within the unroll.
        old_build = agent._build

        @dynamic_batching.batch_fn
        def build(*args):
          with tf.device('/gpu'):
            return old_build(*args)
        tf.logging.info('Using dynamic batching.')
        agent._build = build

    # Build actors and ops to enqueue their output.
    enqueue_ops = []
    for actor_i in range(FLAGS.num_actors):
      if is_actor_fn(actor_i):
        env = create_environment(env_sampler, seed=actor_i+1)
        tf.logging.info('Creating actor %d with level %s',
                        actor_i, actor_task_name_params['task_name'][actor_i])
        actor_output = build_actor(
            agent, env, actor_task_name_params['var'][actor_i].read_value(), action_set)
        with tf.device(shared_job_device):
          enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))

    # Build evaluation ops for every task, which will keep computing returns
    # on all tasks.
    evaluation_output = {}
    if is_learner:
      with tf.name_scope("evaluation"):
        for task_name in task_names:
          env = create_environment(
              env_sampler, initial_task_name=task_name, seed=1)
          evaluation_output[task_name] = build_actor(
              agent, env, task_name, action_set)

    # If running in a single machine setup, run actors with QueueRunners
    # (separate threads).
    if is_learner and enqueue_ops:
      tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    # Build learner.
    if is_learner:
      # Create global step, which is the number of environment frames processed.
      tf.get_variable(
          'num_environment_frames',
          initializer=tf.zeros_initializer(),
          shape=[],
          dtype=tf.int64,
          trainable=False,
          collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

      # Create batch (time major) and recreate structure.
      dequeued = queue.dequeue_many(FLAGS.batch_size)
      dequeued = nest.pack_sequence_as(structure, dequeued)

      def make_time_major(s):
        return nest.map_structure(
            lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)

      dequeued = dequeued._replace(
          env_outputs=make_time_major(dequeued.env_outputs),
          agent_outputs=make_time_major(dequeued.agent_outputs))

      with tf.device('/gpu'):
        # Using StagingArea allows us to prepare the next batch and send it to
        # the GPU while we're performing a training step. This adds up to 1 step
        # policy lag.
        flattened_output = nest.flatten(dequeued)
        area = tf.contrib.staging.StagingArea(
            [t.dtype for t in flattened_output],
            [t.shape for t in flattened_output])
        stage_op = area.put(flattened_output)

        data_from_actors = nest.pack_sequence_as(structure, area.get())

        # Unroll agent on sequence, create losses and update ops.
        done, infos, num_env_frames_and_train, progress_signal = (
            build_learner(agent, data_from_actors.agent_state,
                          data_from_actors.env_outputs,
                          data_from_actors.agent_outputs, teacher_task_ph))

    # Create MonitoredSession (to run the graph, checkpoint and log).
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
    config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
    with tf.train.MonitoredTrainingSession(
            server.target,
            is_chief=is_learner,
            checkpoint_dir=FLAGS.logdir,
            save_checkpoint_secs=600,
            save_summaries_secs=30,
            log_step_count_steps=50000,
            config=config,
            hooks=[py_process.PyProcessHook()]) as session:

      if is_learner:
        summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

        # Prepare data for first run.
        session.run_step_fn(
            lambda step_context: step_context.session.run(stage_op))

        # Execute learning and track performance.
        num_env_frames_v = 0
        num_teacher_update = 0
        next_task_switch_at = FLAGS.switch_tasks_every_k_frames
        last_return_tasks = collections.defaultdict(float)
        task_average_returns = collections.defaultdict(float)
        advantage_previous_returns = collections.defaultdict(float)
        progress_since_switch = []
        returns_task_since_switch = collections.defaultdict(list)
        teacher_history = collections.defaultdict(dict)
        evaluation_task_returns = collections.defaultdict(float)
        next_evaluation_at = FLAGS.evaluate_every_k_frames

        teacher_selected_task_name = actor_task_name_params['task_name'][0]

        while num_env_frames_v < FLAGS.total_environment_frames:
          # Perform one training step, on a minibatch.
          (done_v, infos_v, num_env_frames_v, progress_signal_v,
           _) = session.run(
               (done, infos,
                num_env_frames_and_train, progress_signal, stage_op),
               feed_dict={
                   teacher_task_ph: teacher_selected_task_name
               })

          # Per task, let's average metrics in the current minibatch.
          for task_name in task_names:
            # Only keep part of the minibatch for the current task.
            done_task = done_v & (infos_v.task_name == task_name)
            if np.any(done_task):
              # This task was present in this minibatch
              task_episode_return = np.mean(infos_v.episode_return[done_task])
              task_episode_frames = np.mean(
                  infos_v.episode_step[done_task] * FLAGS.num_action_repeats)

              if task_name == teacher_selected_task_name:
                # Keep the progress_signal across training batches.
                # Only do so if the task corresponds to what the Teacher asked.
                # This will discard progress_signal_v for minibatches that have
                # old tasks.
                progress_since_switch.append(progress_signal_v)

              # For every task, keep the last returns.
              last_return_tasks[task_name] = task_episode_return

              # One summary per task in this minibatch.
              summary = tf.summary.Summary()
              summary.value.add(
                  tag=task_name + '/episode_return',
                  simple_value=task_episode_return)
              summary.value.add(
                  tag=task_name + '/episode_frames',
                  simple_value=task_episode_frames)
              summary.value.add(
                  tag=task_name + '/progress',
                  simple_value=progress_signal_v)
              summary.value.add(
                  tag='Teacher/progress_signal_' + FLAGS.progress_signal,
                  simple_value=progress_signal_v)
              summary.value.add(
                  tag='Teacher/task_selected',
                  simple_value=task_names.index(task_name))
              summary_writer.add_summary(summary, num_env_frames_v)

            # Keep track of returns for all tasks, through time
            # (default to 0 if the task was never selected yet)
            # This will keep the last score even when the task is not retrained
            # on, but that's actually what Tensorboard shows, so it's ok.
            returns_task_since_switch[task_name].append(
                last_return_tasks[task_name])

          # Perform a full evaluation on all tasks
          if num_env_frames_v >= next_evaluation_at:
            summary_evaluator = tf.summary.Summary()

            for task_name in task_names:
              returns = []
              while len(returns) < FLAGS.test_num_episodes:
                rewards_v, done_v = session._tf_sess().run(
                    (evaluation_output[task_name].env_outputs.reward,
                     evaluation_output[task_name].env_outputs.done))

                # Repack the environment outputs
                rewards_v = rewards_v[1:]
                done_v = done_v[1:]

                # Check the performance
                episode_returns = rewards_v[done_v]
                returns.extend(episode_returns)

              # Store mean returns per task
              returns_avg = np.mean(returns)
              evaluation_task_returns[task_name] = returns_avg

              # Logging/Tensorboard
              tf.logging.info('[%d] Evaluating task %s -> episode return: %f',
                              num_env_frames_v, task_name, returns_avg)
              summary_evaluator.value.add(
                  tag='Evaluation/' + task_name + '/episode_return',
                  simple_value=returns_avg)

              # Also use these evaluation values to bootstrap the Advantage
              # previous rewards
              advantage_previous_returns[task_name] = returns_avg

            summary_writer.add_summary(summary_evaluator, num_env_frames_v)
            next_evaluation_at += FLAGS.evaluate_every_k_frames

          # Now ask the Teacher for new tasks to train on!
          if num_env_frames_v >= next_task_switch_at:
            print("Let's update the tasks for all actors now!")

            # Compute average return for ~all tasks since last switch
            task_average_returns = {
                task_name: np.mean(returns_task_since_switch[task_name])
                for task_name in task_names
            }

            # Compute the progress signal for the Teacher
            if FLAGS.progress_signal == 'advantage':
              # For the Advantage (reward[T] - reward[T-K]), we need to compare
              # to "previous" reward values.
              # Previous rewards are either evaluation_task_returns or
              # task_average_returns, whichever is "fresher"
              rewards_post_switch = np.mean(progress_since_switch or 0)
              progress_for_teacher = (
                  rewards_post_switch -
                  evaluation_task_returns[teacher_selected_task_name])

              # Update last returns
              # advantage_previous_returns[
              # teacher_selected_task_name] = rewards_post_switch
            else:
              # For the other signals, we can use them directly.
              progress_for_teacher = np.mean(progress_since_switch or 0)

            # Update Teacher according to the progress signal we got!
            teacher.update(teacher_selected_task_name, progress_for_teacher)

            # Log / Tensorboard
            tf.logging.info("[%d][%d] Task: %s, Episode return mean: %.3f, "
                            "Teacher progress signal: %.3f",
                            num_teacher_update, num_env_frames_v,
                            teacher_selected_task_name,
                            task_average_returns[teacher_selected_task_name],
                            progress_for_teacher)
            summary_teacher = tf.summary.Summary()
            summary_teacher.value.add(
                tag='Teacher/at_update_task_returns',
                simple_value=task_average_returns[teacher_selected_task_name])
            summary_teacher.value.add(
                tag='Teacher/at_update_progress_signal',
                simple_value=progress_for_teacher)
            summary_writer.add_summary(summary_teacher, num_env_frames_v)

            # Keep track of teacher state
            teacher_history['progress_signal'][num_teacher_update] = (
                progress_for_teacher)
            teacher_history['weights'][num_teacher_update] = (
                teacher._log_weights.copy())
            teacher_history['arm_probs'][num_teacher_update] = (
                teacher.task_probabilities.copy())
            teacher_history['teacher_selected_task_name'][num_teacher_update] = (
                teacher_selected_task_name)
            teacher_history['num_env_frames'][num_teacher_update] = (
                num_env_frames_v)
            teacher_history['task_returns'][num_teacher_update] = (
                task_average_returns)
            teacher_history['evaluation_task_returns'][num_teacher_update] = (
                evaluation_task_returns.copy())
            teacher_history['task_names'] = task_names

            # Store teacher history for analysis
            if ((num_teacher_update + 1) %
                FLAGS.save_every_k_teacher_updates == 0):
              np.save(
                  os.path.join(FLAGS.logdir, "teaching_output_{}.npy".format(
                      num_teacher_update)),
                  dict(teacher_history))
              # Reset teacher history to be super safe
              teacher_history = collections.defaultdict(dict)

            # Get new task from the Teacher and update Actors
            teacher_selected_task_name = teacher.get_task()
            update_all_actors_tasks(
                [teacher_selected_task_name],
                actor_task_name_params,
                session._tf_sess(),
                single_task=True)

            # ... finish this switch
            progress_since_switch = []
            returns_task_since_switch = collections.defaultdict(list)
            num_teacher_update += 1
            next_task_switch_at += FLAGS.switch_tasks_every_k_frames
            print("Switching to task {}! Next update at {}".format(
                teacher_selected_task_name, next_task_switch_at))

      else:
        # Execute actors (they just need to enqueue their output).
        while True:
          session.run(enqueue_ops)


def test(action_set):
  """Test."""
  with tf.Graph().as_default():
    # Get EnvironmentFactory
    env_sampler = env_factory.EnvironmentFactory(
        FLAGS.recipes_path, FLAGS.hints_path, max_steps=FLAGS.max_steps,
        reuse_environments=FLAGS.reuse_environments, seed=1, visualise=True)
    dummy_env = env_sampler.sample_environment()
    obs_spec = dummy_env.obs_specs()
    task_names = sorted(env_sampler.task_names)

    agent = Agent(len(action_set), obs_spec)
    outputs = {}
    task_returns = collections.defaultdict(lambda: collections.deque([], 5000))

    # Test on all environments one after another
    for task_name in task_names:
      env = create_environment(
          env_sampler, initial_task_name=task_name, seed=1)
      outputs[task_name] = build_actor(agent, env, task_name, action_set)

    with tf.train.SingularMonitoredSession(
            checkpoint_dir=FLAGS.logdir,
            hooks=[py_process.PyProcessHook()]) as session:
      for task_name in task_names:
        tf.logging.info('Testing task: %s', task_name)
        while True:
          rewards_v, done_v, observations_v = session.run(
              (outputs[task_name].env_outputs.reward,
               outputs[task_name].env_outputs.done,
               outputs[task_name].env_outputs.observation))

          # Repack the environment outputs
          rewards_v = rewards_v[1:]
          done_v = done_v[1:]
          observations_dict = {
            obs_name: observations_v[obs_i][1:]
            for obs_i, obs_name in enumerate(obs_spec.keys())
          }

          # Check the performance
          episode_returns = rewards_v[done_v]
          returns = task_returns[task_name]
          returns.extend(episode_returns)

          # Visualise render
          for frame_i, frame in enumerate(observations_dict['image']):
            dummy_env.render_matplotlib(frame=frame, delta_time=0.05)
            if rewards_v[frame_i]:
              rewarding_frame = np.ones_like(frame) * np.array([0, 1, 0])
              dummy_env.render_matplotlib(
                  frame=rewarding_frame, delta_time=0.2)

          if len(returns) >= FLAGS.test_num_episodes:
            tf.logging.info('Mean episode return: %f', np.mean(returns))
            break


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  action_set = environments.DEFAULT_ACTION_SET
  if FLAGS.mode == 'train':
    train(action_set)
  else:
    test(action_set)


if __name__ == '__main__':
  tf.app.run()
