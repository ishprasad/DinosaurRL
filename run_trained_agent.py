import pygame
import torch
import argparse
import os
from env import DinoEnv
from rl_agent import DQN, get_state

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400

def run_policy(policy_file, render_speed=60):
    env = DinoEnv()
    state_dim = 5  
    action_dim = 2
    actions = ["nothing", "jump"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dim).to(device)
    
    if not os.path.exists(policy_file):
        print(f"Policy file {policy_file} not found!")
        return
    policy_net.load_state_dict(torch.load(policy_file, map_location=device))
    policy_net.eval()
    
    env.reset()
    state = get_state(env)
    total_reward = 0
    step = 0
    
    print(f"Running policy from {policy_file}...")
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print(f"Episode terminated. Score: {env.score:.2f}, Total Reward: {total_reward:.2f}, Steps: {step}")
                return
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        action_idx = q_values.argmax().item()
        action = actions[action_idx]
        
        env.step(action)
        
        next_state = get_state(env)
        reward = 0.1 
        if env.game_over:
            reward = -100
        done = env.game_over
        
        state = next_state
        total_reward += reward
        step += 1
        
        if done:
            print(f"Episode finished. Score: {env.score:.2f}, Total Reward: {total_reward:.2f}, Steps: {step}")
            break
        
        env.clock.tick(render_speed)
    
    pygame.quit()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run a trained DQN policy for the Dino Game.")
    parser.add_argument("--policy_file", type=str, required=True, help="Path to the .pt policy file")
    parser.add_argument("--render_speed", type=int, default=60, help="Frames per second for rendering")
    args = parser.parse_args()
    
    # Run the policy
    run_policy(args.policy_file, args.render_speed)

if __name__ == "__main__":
    main()