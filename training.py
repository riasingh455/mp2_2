#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import argparse
import os
import time
from new_game import Game

def train_agent(num_episodes=10000, fps=1000, save_interval=100):
    """
    Train the agent through multiple game simulations
    
    Args:
        num_episodes: Number of simulations to execute
        fps: Game speed (frames per second)
        save_interval: How often to print and save the state (in episodes)
    """
    # Record start time
    start_time = time.time()
    
    # Record all scores
    all_scores = []
    
    # Create results directory
    os.makedirs('training_results', exist_ok=True)
    
    # Perform multiple simulations
    for episode in range(1, num_episodes + 1):
        # Generate different random seeds for each simulation
        seed = random.randint(1, 1000000)
        # print(seed)
        
        # Randomize number of entities for each episode
        num_zombies = random.randint(10, 30)
        num_creepers = random.randint(5, 15)
        num_skeletons = random.randint(5, 15)
        num_chests = random.randint(10, 20)
        num_barrels = random.randint(10, 20)
        
        # Create game instance with random parameters
        game = Game(
            width=50,  # default width
            height=30,  # default height
            initial_energy=1000,  # default energy
            grid_size=24,  # default grid size
            fog=True,  # default fog setting
            num_zombies=num_zombies,  # random zombies (10-30)
            num_creepers=num_creepers,  # random creepers (5-15)
            num_skeletons=num_skeletons,  # random skeletons (5-15)
            p_gold=0.2,  # default gold probability in stone
            p_dgold=0.4,  # default gold probability in deepslate
            num_chests=num_chests,  # random chests (10-20)
            num_barrels=num_barrels,  # random barrels (10-20)
            training=True,  # enable training
            render=False  # turn off rendering
        )
        
        # Run the game and get the final score
        # print(f"Episode {episode}/{num_episodes}, Seed: {seed}")
        final_score = game.run(fps)
        all_scores.append(final_score)
        
        # Save results and print statistics every save_interval episodes
        if episode % save_interval == 0:
            # Calculate current average score and highest score
            avg_score = sum(all_scores[-save_interval:]) / save_interval
            max_score = max(all_scores[-save_interval:])
            elapsed_time = time.time() - start_time
            
            # Print statistics
            print(f"Episodes: {episode}/{num_episodes}")
            print(f"Average Score (last {save_interval}): {avg_score:.2f}")
            print(f"Max Score (last {save_interval}): {max_score}")
            print(f"Time Elapsed: {elapsed_time:.2f} seconds")
            print("-------------------------------")
            
            # Save score records to file
            with open(os.path.join('training_results', 'scores.txt'), 'a') as f:
                for score in all_scores[-save_interval:]:
                    f.write(f"{score}\n")
    
    # Training complete, calculate overall statistics
    total_time = time.time() - start_time
    avg_score_total = sum(all_scores) / len(all_scores)
    max_score_total = max(all_scores)
    
    # Print overall statistics
    print("\n===== Training Complete =====")
    print(f"Total Episodes: {num_episodes}")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Score: {avg_score_total:.2f}")
    print(f"Max Score: {max_score_total}")
    
    # Save overall statistics to file
    with open(os.path.join('training_results', 'summary.txt'), 'w') as f:
        f.write(f"Total Episodes: {num_episodes}\n")
        f.write(f"Total Training Time: {total_time:.2f} seconds\n")
        f.write(f"Average Score: {avg_score_total:.2f}\n")
        f.write(f"Max Score: {max_score_total}\n")
    
    return all_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Minecraft Mining agent.")
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes to train')
    parser.add_argument('--fps', type=int, default=1000, help='Frames per second for the game')
    parser.add_argument('--save_interval', type=int, default=100, help='Save and print stats every N episodes')
    
    args = parser.parse_args()
    
    # Set random seed to ensure reproducibility
    random.seed(42)
    
    # Start training
    train_agent(
        num_episodes=args.episodes,
        fps=args.fps,
        save_interval=args.save_interval
    )