# AcceleratedRL
Accelerated Reinforcement Learning via Imitation Learning

## Dependencies
See `dl4cv.yml` for dependencies information

## Objectives
In a sparse reward environment setting with large state and action space, it can be hard to stumble upon a favorable policy for a reinforcement learning agent. To tackle this problem, we employ imitation learning to first learn human-inspired policy to kickstart the reinforcement learning process. To demonstrate that using imitation learning shortens the training time in comparison to only using reinforcement learning, we will present two graphs showing how the reinforcement learning agent augmented with imitation learning learns much faster than the vanilla RL agent does.
We are training an agent that plays Atari and Nintendo games using OpenAI Retro framework. Imitation learning and reinforcement learning are well-established in literature, but combining the two to accelerate the learning is our original idea.
Here are some initial proposal of our model architecture:
- Imitation Learning (Tensorflow):
  - Training data: human game-play data in frames of pixels (states)
  - Output: action to take in the state
  - Architecture:
    - Conv
    - Max-Pooling
    - Conv
    - Max-Pooling
    - Fully-Connected
    - Fully-Connected
- Reinforcement Learning:
  - State: agent game-play data in frames of pixels
  - Action: the current policy’s decision
  - Reward: depending on games
  - Algorithm: PPO2 in OpenAI baselines
