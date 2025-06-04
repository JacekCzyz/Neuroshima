from neuro_env_dqn_minmax import NeuroHexEnv as NeuroHexEnv_dqn
from neuro_env_ppo_minmax import NeuroHexEnv as NeuroHexEnv_ppo
import neuro_env_dqn_minmax
import neuro_env_ppo_minmax
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

    env = NeuroHexEnv_ppo()
    env = ActionMasker(env, neuro_env_ppo_minmax.mask_fn)
    
    model = MaskablePPO.load("neuroshima_ppo_model_400-500-000_minmax_vs_same_fix.zip", env = env)

    obs = env.env.reset()[0]
    clock = pygame.time.Clock()
    results=[0,0,0]
    
    test_result_file = open("ppo_test_results_selfplay_vs_self.csv", "w")
    
    test_result_file.write("neuroshima_ppo_model_400-000_minmax_vs_same_fix.zip\n")

    for i in range(1000):
        done=False
        while not done:
            clock.tick(200)
            check_battle = True
            input("Press enter to play a game")
            if not env.env.turn_started:
                map_utils.fill_choice(env.env.choice, env.env.current_player, env.env.first_turn, False)
                env.env.turn_started = True

            if env.env.current_player == 0:
                if len(env.env.choice[0]) > 0:
                    move_time_start = time.time()
                    


                    with torch.no_grad():   #ppo
                        obs_tensor = obs_as_tensor(obs, model.policy.device)
                        obs_tensor = obs_tensor.unsqueeze(0)
                        action_masks = get_action_masks(env)
                        action, value, log_prob = model.policy.forward(obs_tensor, action_masks=action_masks)
                        action = action.item()
                    obs, reward, done, info = env.env.step(action)
                    
                    move_time_end = time.time()
                    time_elapsed = move_time_end - move_time_start
                    test_result_file.write(f"{time_elapsed:.4f}, {reward}; ")
                    if done:
                        obs = env.env.reset()[0]

            if any(hex.skin.team == 0 for row in env.env.map.hex_row_list for hex in row.hex_list):
                check_battle = False
    
            if check_battle:
                env.env.Player1hp, env.env.Player2hp = map_utils.battle(env.env.map, env.env.Player1hp, env.env.Player2hp)  
            
            
            if env.env.Player1hp <= 0 or env.env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
                final_reward = 0
                if env.env.Player1hp > env.env.Player2hp:
                    final_reward = 1.0
                    results[0]+=1
                    test_result_file.write(f"{final_reward}"+"\n"+"1won" + "\n")
                    
                elif env.env.Player1hp < env.env.Player2hp:
                    final_reward = -1.0
                    results[1]+=1     
                    test_result_file.write(f"{final_reward}"+"\n"+"2won" + "\n")
                            
                else:
                    results[2]+=1                
                    final_reward = 0.0
                    test_result_file.write(f"{final_reward}"+"\n"+"tie" + "\n")
                              
                print("Player1 won" if env.env.Player1hp > env.env.Player2hp else "Player2 won" if env.env.Player2hp > env.env.Player1hp else "TIE!!!")
                obs = env.env.reset()[0]
                continue
                        
            
            check_battle = True
            
            if env.env.current_player == 1:
                map_utils.fill_choice(env.env.choice, env.env.current_player, env.env.first_turn, False)
                env.env.turn_started = True
                if len(env.env.choice[1]) > 1 or env.env.first_turn:
                    env.env.Player1hp, env.env.Player2hp = minmax.min_max(env.env.map, env.env.choice, 1, env.env.Player1hp, env.env.Player2hp, 0)
                      
                env.env.current_player = 0
                env.env.turn_started = False
                if env.env.first_turn:
                    env.env.first_turn = False

            if any(hex.skin.team == 0 for row in env.env.map.hex_row_list for hex in row.hex_list):
                check_battle = False
 
            if check_battle:
                env.env.Player1hp, env.env.Player2hp = map_utils.battle(env.env.map, env.env.Player1hp, env.env.Player2hp)

            env.env.render()
            if env.env.Player1hp <= 0 or env.env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
                final_reward = 0
                if env.env.Player1hp > env.env.Player2hp:
                    final_reward = 1.0
                    results[0]+=1
                    test_result_file.write(f"{final_reward}"+"\n"+"1won" + "\n")
                    
                elif env.env.Player1hp < env.env.Player2hp:
                    final_reward = -1.0
                    results[1]+=1     
                    test_result_file.write(f"{final_reward}"+"\n"+"2won" + "\n")
                            
                else:
                    results[2]+=1                
                    final_reward = 0.0
                    test_result_file.write(f"{final_reward}"+"\n"+"tie" + "\n")
                              
                print("Player1 won" if env.env.Player1hp > env.env.Player2hp else "Player2 won" if env.env.Player2hp > env.env.Player1hp else "TIE!!!")
                obs = env.env.reset()[0]


    test_result_file.write(f"Player1 won: {results[0]}\n")
    test_result_file.write(f"Player2 won: {results[1]}\n")
    test_result_file.write(f"ties: {results[2]}\n")
    test_result_file.close()
    
