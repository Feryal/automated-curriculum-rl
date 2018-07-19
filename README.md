# Automated Curriculum Learning

How would you make an agent capable of solving the complex hierarchical tasks?

Imagine a problem that is complex and requires a collection of skills, which are extremely hard to learn in one go with sparse rewards (e.g. solving the shape sorter in robotics). Hence, one might need to learn to generate a curriculum of simpler tasks, so that overall a student network can learn to perform a complex task efficiently. This could be set up as a Teacher network which keeps track of the capabilities that the student is currently possessing, and proposes new tasks when it is time to do so.

In order to start, I adapted the Craft Environment from work by Andreas et al.,[[1]](##References) as it has a nice simple structure with possibility of hierarchical task design with a range of complexity that is fast to iterate through. I have developed a fixed curriculum of simpler target sub-tasks (in craft environment: "get wood" "get grass" "get iron" "make cloth" "get gold"), and in the future will make a teacher network who proposes tasks for the student to learn. I could also kick-start the student with demonstrations from an expert.

Currently, I have interfaced IMPALA[[2]](##References), a GPU utilised  version of A3C architecture which uses multiple distributed actors with V-Trace off-policy correction, with my Craft Environment to train on all the possible Craft tasks concurrently. This is possible by providing the hash of the task name as instruction to the network (similar setup to DMLab IMPALA, using an LSTM to process the instruction, maybe an overkill here). I am currently investigating the effect of hyper parameters (e.g. episode length) and network architecture on performance. These tasks are currently randomly selected and proposed to the student network with a hyper parameter for how often the task is changed for each actor. Note, that every episode the environment setup changes for a given task. Next, I will implement my teacher network in order to have a smarter, automated, task proposal.

Other papers that I am inspired by in this work include [[3]](##References), [[4]](##References).
<div style="text-align:center ; width:200px;">
<a href='https://xkcd.com/894/'>

![](https://www.explainxkcd.com/wiki/images/6/69/progeny.png)

</a>
</div>

## Usage:

Run with

```sh
python experiment.py --num_actors=48 --batch_size=32
```

## Dependencies:


- [Python 2.7](https://www.python.org/)
- [NumPy](http://www.numpy.org/)
- [tf-nightly==1.9.0-dev20180530]()
- [dm-sonnet]()

## Note:

I have done multiple modifications to the Craft Environment and IMPALA code for integration (will document later). This code should also be able to integrate with Gym environments with minor changes, which will be added soon. Currently the wrapper for the Craft Env mimics [DMLab](https://github.com/deepmind/lab) interface.

## Results

TBA

## Acknowledgements

- [@DeepMind](https://github.com/deepmind) for open-sourcing GPU accelerated distributed RL algorithm capable of multitask learning [IMPALA](https://github.com/deepmind/scalable_agent)
- [@jacobandreas](https://github.com/jacobandreas) for open-sourcing the mine-craft inspired Craft Environment used in Policy Sketches paper [[1]](##References) [Craft Environment](https://github.com/jacobandreas/psketch)


## References

* [1] [Modular Multitask Reinforcement Learning with Policy Sketches](https://arxiv.org/abs/1611.01796) (Andreas et al., 2016)
* [2] [Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003) (Graves et al., 2017)
* [3] [Learning by Playing-Solving Sparse Reward Tasks from Scratch](https://arxiv.org/abs/1802.10567) (Reidmiller et al., 2018)
* [4] [POWERPLAY: Training an Increasingly General Problem Solver by Continually Searching for the Simplest Still Unsolvable Problem](https://arxiv.org/abs/1602.01783) (Schmidhuber, 2011)
