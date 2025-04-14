import pygame
from env import DinoEnv
from rl_agent import DQNAgent, get_state

episodes=901
env_game = DinoEnv()
state_dim = 5  # [dino_y, dino_velocity, obs_distance, obs_y, obs_type]
action_dim = 2  # 0: do nothing, 1: jump
agent = DQNAgent(state_dim, action_dim)
actions = ["nothing", "jump"]
    
for episode in range(episodes):
    env_game.reset()
    state = get_state(env_game)
    total_reward = 0
    step = 0
        
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            
        action_idx = agent.select_action(state)
        action = actions[action_idx]
        env_game.step(action)
            
        next_state = get_state(env_game)
        reward = 0.1 
        if env_game.game_over:
            reward = -100 
        done = env_game.game_over
            
        agent.memory.push((state, action_idx, reward, next_state, done))
            
        state = next_state
        total_reward += reward
        step += 1
            
        agent.optimize()
            
        if done:
            break
            
        env_game.clock.tick(2000)
        
    if episode % 10 == 0:
        agent.update_target_net()
    if episode % 100 == 0:
        agent.save_policy(episode)
        
    print(f"Episode {episode}/{episodes}, Score: {env_game.score:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
pygame.quit()

