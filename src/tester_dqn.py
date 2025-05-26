from neuro_env_dqn_minmax import NeuroHexEnv as NeuroHexEnv_dqn
import neuro_env_dqn_minmax
import pygame
import map_utils
import skins
from stable_baselines3 import DQN
import minmax
import time 
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import obs_as_tensor
import torch

if __name__ == "__main__":
    env = NeuroHexEnv_dqn()
    model = DQN.load("neuroshima_dqn_new_selfplay_1-500-000_model_vs_same")


    obs = env.reset()[0]
    clock = pygame.time.Clock()
    results=[0,0,0]
    
    test_result_file = open("dqn_test_results_selfplay_vs_self.csv", "w")
    
    test_result_file.write("dqn_test_results_selfplay_vs_self.zip\n")

    for i in range(1000):
        done=False
        while not done:
            clock.tick(30)
            #input("Press enter to play a game")
            
            check_battle = True
            if not env.turn_started:
                map_utils.fill_choice(env.choice, env.current_player, env.first_turn, False)
                env.turn_started = True

            if env.current_player == 0:
                if len(env.choice[0]) > 0:
                    move_time_start = time.time()
                    
                    epsilon = model.exploration_rate
                    action = neuro_env_dqn_minmax.masked_predict(model, obs, env, epsilon=epsilon) #dqn
                    obs, reward, done, truncated, info = env.step(action)
                   
                    move_time_end = time.time()
                    time_elapsed = move_time_end - move_time_start
                    test_result_file.write(f"{time_elapsed:.4f}, {reward}; ")
                    if done:
                        obs = env.reset()[0]

            if any(hex.skin.team == 0 for row in env.map.hex_row_list for hex in row.hex_list):
                check_battle = False
    
            if check_battle:
                env.Player1hp, env.Player2hp = map_utils.battle(env.map, env.Player1hp, env.Player2hp)  
            
            
            if env.Player1hp <= 0 or env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
                final_reward = 0
                if env.Player1hp > env.Player2hp:
                    final_reward = 1.0
                    results[0]+=1
                    test_result_file.write(f"{final_reward}"+"\n"+"1won" + "\n")
                    
                elif env.Player1hp < env.Player2hp:
                    final_reward = -1.0
                    results[1]+=1     
                    test_result_file.write(f"{final_reward}"+"\n"+"2won" + "\n")
                            
                else:
                    results[2]+=1                
                    final_reward = 0.0
                    test_result_file.write(f"{final_reward}"+"\n"+"tie" + "\n")
                              
                print("Player1 won" if env.Player1hp > env.Player2hp else "Player2 won" if env.Player2hp > env.Player1hp else "TIE!!!")
                obs = env.reset()[0]
                continue
                        
            
            check_battle = True
            
            if env.current_player == 1:
                map_utils.fill_choice(env.choice, env.current_player, env.first_turn, False)
                env.turn_started = True
                if len(env.choice[1]) > 1 or env.first_turn:
                    env.Player1hp, env.Player2hp = minmax.min_max(env.map, env.choice, 1, env.Player1hp, env.Player2hp, 0)
                      
                env.current_player = 0
                env.turn_started = False
                if env.first_turn:
                    env.first_turn = False

            if any(hex.skin.team == 0 for row in env.map.hex_row_list for hex in row.hex_list):
                check_battle = False
 
            if check_battle:
                env.Player1hp, env.Player2hp = map_utils.battle(env.map, env.Player1hp, env.Player2hp)

            env.render()
            if env.Player1hp <= 0 or env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
                final_reward = 0
                if env.Player1hp > env.Player2hp:
                    final_reward = 1.0
                    results[0]+=1
                    test_result_file.write(f"{final_reward}"+"\n"+"1won" + "\n")
                    
                elif env.Player1hp < env.Player2hp:
                    final_reward = -1.0
                    results[1]+=1     
                    test_result_file.write(f"{final_reward}"+"\n"+"2won" + "\n")
                            
                else:
                    results[2]+=1                
                    final_reward = 0.0
                    test_result_file.write(f"{final_reward}"+"\n"+"tie" + "\n")
                              
                print("Player1 won" if env.Player1hp > env.Player2hp else "Player2 won" if env.Player2hp > env.Player1hp else "TIE!!!")
                obs = env.reset()[0]


    test_result_file.write(f"Player1 won: {results[0]}\n")
    test_result_file.write(f"Player2 won: {results[1]}\n")
    test_result_file.write(f"ties: {results[2]}\n")
    test_result_file.close()
    
