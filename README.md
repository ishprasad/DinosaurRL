# DinosaurRL
A deep reinforcement learning agent that is trained to play an imitation of the Google Chrome Dinosaur Game.
## Demonstration
The agent learns to play the game perfectly after 800 episodes of training.
![](demo.gif)
## Requirements
- pygame
- numppy
- torch
## Running
- Training the agent for 901 episodes (the hyperparameters in rl_agent.py and the number of episodes in run_training.py)

`python run_training.py`
- Watching how a trained agent performs

`python run_trained_agent.py --policy_file trained_policy.pt --render_speed 120`
- Playing the game as a user

`python run_user_play.py`
## How it works
The Dinosaur Game, usually seen on Google Chrome when the internet is unavailable, is a side-scrolling game in which the user controls a dinosaur, making it jump or crouch to avoid obstacles (cacti and pterodactyls). This project involves a reinforcement learning (RL) agent that learns to play an imitation of the game. In this imitation, among other differences, the shape of the agent (dinosaur) and obstacles are mere rectangles (as shown in the demo above), and the agent has only two possible actions (jump or do nothing).

As an RL agent, the agent begins taking random actions, measuring the rewards it gets as it passes through different states. As the agent's goal is to survive as long as it can, maximizing its score, it gets a reward, +0.1, for each step without colliding with an obstacle and a penalty, -100, for ending the game by colliding. The agent begins behaving exploratorily (where episilon (the probability of choosing a random action) is 1). Over time, as it learns which action in a given state will maximize its total reward for the episode/run, it behaves more exploitatively or greedily (where episilon decreases (meaning the probability of choosing an action that maximizes reward increases)).

To learn the best action for a given

#This is a WIP


