import gym
from gym import spaces
import torch
import numpy as np
import pygame
import map_utils
from hex import hexagon_map
from skins import tile
import skins
from pygame import font
from collections import deque
import random
from stable_baselines3 import DQN
import minmax

class NeuroHexEnv(gym.Env):
    def __init__(self):
        super(NeuroHexEnv, self).__init__()
        
        self.action_space = spaces.Discrete(342) #each 114 actions mean-> xth tile into yth tile with zth rotation

        self.observation_space = spaces.Box(low=0, high=255, shape=(100,), dtype=np.float32)
        self.choice = [[], []]
        self.screen = None
        self.map = None
        self.current_player = 0
        self.Player1hp, self.Player2hp = 20, 20
        self.done = False
        self.first_turn = True
        self.turn_started = False        
        self.reset()

    def reset(self):
        self.choice = [[], []]
        self.screen = None
        self.map = None
        self.current_player = 0
        self.Player1hp, self.Player2hp = 20, 20
        self.done = False
        self.first_turn = True
        self.turn_started = False
        map_utils.team_tiles = map_utils.team_tiles_safe_copy.copy() 
        return self._get_obs()

    def step(self, action):
        reward = 0
        done = False
        chosen_tile = 0
        place_index = 0
        rotations = 0
        if action < 114:
            chosen_tile = 0
            place_index = action / 19
            rotations = action % 19
        elif action < 228:
            temp = action - 114
            chosen_tile = 1
            place_index = temp / 19
            rotations = temp % 19            
        else:
            temp = action - 228
            chosen_tile = 2
            place_index = temp / 19
            rotations = temp % 19   
        battle=False
        selected_skin = self.choice[0][chosen_tile].skin
        if selected_skin.team == 3:
            battle=True
            self.Player1hp, self.Player2hp = map_utils.battle(self.map, self.Player1hp, self.Player2hp)
            self.choice[0].pop(chosen_tile)
        else:
            for i in range(rotations):
                selected_skin.rotate(1)
            q,r=0,0
            if place_index<=2:
                q=0
                r=place_index
            elif place_index<=6:
                q=1
                r=place_index-3
            elif place_index<=11:
                q=2
                r=place_index-7
            elif place_index<=15:
                q=3
                r=place_index-12
            else:
                q=4
                r=place_index-16
            
            for i in range(len(self.map.free_hexes)):
                hex = self.map.free_hexes[i]
                if hex.q==q and hex.r==r:
                    hex.skin = tile(selected_skin.color, selected_skin.close_attack, selected_skin.ranged_attack,
                                                selected_skin.team, selected_skin.initiative, selected_skin.lives, selected_skin.hq)
                    self.map.taken_hexes.append(hex)
                    self.map.free_hexes.pop(i)                            
                    self.choice[0].pop(chosen_tile)
                                   
                    if len(self.choice[0])==1:
                        self.choice[0].pop(0)
                        self.current_player = 1
                        self.turn_started = False  
                    break               

        enemy_attacked = 2
        enemy_hq_attacked = 4
        ally_hq_attacked = -5
        enemy_killed = 2
        ally_killed = -2
        placed_hex_attacked=-2
        if battle:
            enemy_attacked = 3
            ally_attacked = -3
            enemy_hq_attacked = 5
            ally_hq_attacked = -6

        attacked_hexes = []
        for hex in self.hex_map.taken_hexes: #first check all tiles
            attacked_hexes.extend(map_utils.get_neighbors(self.hex_map, hex.q, hex.r))
            attacked_hexes.extend(map_utils.get_neighbor_lines(self.hex_map, hex.q, hex.r))
        
        checked_state = []
        for hex in attacked_hexes:
            if hex.skin.team==1:
                if hex.skin.hq:
                    map_value+=ally_hq_attacked
                elif hex.q == q and hex.r == r:
                    map_value+=placed_hex_attacked
                else:
                    map_value+=ally_attacked
                    
                if attacked_hexes.count(hex) >= hex.skin.lives and hex not in checked_state:
                    map_value+=ally_killed  
                    checked_state.append(hex)

            else:
                if hex.skin.hq:
                    map_value+=enemy_hq_attacked
                else:
                    map_value+=enemy_attacked
                    
                if attacked_hexes.count(hex) >= hex.skin.lives and hex not in checked_state:     
                    map_value+=enemy_killed  
                    checked_state.append(hex)

        #additional placed hex value
        attacked_hexes = []
        if not battle:
            attacked_hexes.extend(map_utils.get_neighbors(self.hex_map, q, r))
            attacked_hexes.extend(map_utils.get_neighbor_lines(self.hex_map, q, r))
        for hex in attacked_hexes:
            if hex.skin.hq:
                map_value+=enemy_hq_attacked
            else:
                map_value+=enemy_attacked
                    
            if attacked_hexes.count(hex) >= hex.skin.lives and hex not in checked_state:     
                map_value+=enemy_killed  
                checked_state.append(hex)                   

        reward = map_value
        if self.Player1hp <= 0 or self.Player2hp <= 0:
            done = True
            reward = 1 if self.Player1hp > self.Player2hp else -1

        return self._get_obs(), reward, done, {} #dict->info -> can return battle info

    def _get_obs(self): #simple observation - hexes on map as flattened array
        obs = np.zeros(19, dtype=np.float32)
        index = 0
        for hex_row in self.map.hex_row_list:
            for hex_tile in hex_row.hex_list:
                if hex_tile.skin.team == 0:
                    obs[index] = 0
                elif hex_tile.skin.team == 1:
                    obs[index] = 1
                elif hex_tile.skin.team == 2:
                    obs[index] = 2
                if hex_tile.skin.hq:
                    obs[index] += 0.5
                index+=1
        return obs

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((700, 700))
        self.screen.fill((0, 0, 0))
        map_utils.draw_map(self.screen, self.map)
        text1 = font.render(f"Player1 hp: {self.Player1hp}", True, (255, 255, 255))
        text2 = font.render(f"Player2 hp: {self.Player2hp}", True, (255, 255, 255))
        self.screen.blit(text1, (500, 200))
        self.screen.blit(text2, (500, 300))
        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()


    def get_action_mask(self):
        action_mask = np.ones(342, dtype=bool)
        start_index = 0
        is_hex_free=[]
        
        for hex in self.map.hex_row_list:
            for hex_tile in hex.hex_list:
                if hex_tile.skin.team == 0:
                    is_hex_free.append(1)
                else:
                    is_hex_free.append(0)
        for i in range(3):
            if i<len(self.choice[0]):
                for j in range(19):
                    if is_hex_free[j] == 0:
                        for k in range(6):
                            action_mask[start_index + j * 6 + k] = False
            else:
                for j in range(19):
                    for k in range(6):
                        action_mask[start_index + j * 6 + k] = False
            start_index += 19 * 6
        return action_mask
    
        
def masked_predict(model, obs, env, epsilon=0.1):
    if np.random.rand() < epsilon:
        valid_actions = np.where(env.get_action_mask())[0]
        return np.random.choice(valid_actions)

    q_values = model.policy.q_net(torch.tensor(obs).float().unsqueeze(0)).detach().numpy()[0]
    mask = env.get_action_mask()
    masked_q_values = np.where(mask, q_values, -np.inf)
    action = np.argmax(masked_q_values)
    return action
    
    
if __name__ == "__main__":
    env = NeuroHexEnv()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.1,
        verbose=1,
    )

    total_timesteps = 100_000
    obs = env.reset()

    tile_counter=0
    for step in range(total_timesteps*2):
        check_battle = True
        if not env.turn_started:
            map_utils.fill_choice(env.choice, env.current_player, env.first_turn, False)
            env.turn_started = True
            
        if env.current_player == 0:    
            epsilon = model.exploration_rate
            action = masked_predict(model, obs, env, epsilon=epsilon)
            
            new_obs, reward, done, info = env.step(action)
            check_battle = info.get('check_battle', False)
            model.replay_buffer.add(obs, action, reward, new_obs, done, info)
            
            obs = new_obs

            if done:
                obs = env.reset()

            if step > model.learning_starts:
                model.train(batch_size=model.batch_size, gradient_steps=1)

            if step % model.target_update_interval == 0:
                model.update_target_network()

        if any(hex.skin.team == 0 for row in env.map.hex_row_list for hex in row.hex_list):
            check_battle = False

        if check_battle:
            env.Player1hp, env.Player2hp = map_utils.battle(env.map, env.Player1hp, env.Player2hp)                

        if env.current_player == 1:
            if len(env.choice[1])>1 or env.first_turn:
                env.Player1hp, env.Player2hp = minmax.min_max(env.map, env.choice, 1, env.Player1hp, env.Player2hp, 1)  
            env.current_player = 0
            env.turn_started = False
            if env.first_turn:
                env.first_turn=False
                
        if any(hex.skin.team == 0 for row in env.map.hex_row_list for hex in row.hex_list):
            check_battle = False

        if check_battle:
            env.Player1hp, env.Player2hp = map_utils.battle(env.map, env.Player1hp, env.Player2hp)
        env.render()
        if env.Player1hp <= 0 or env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
            print("Player1 won" if env.Player1hp > env.Player2hp else "Player2 won" if env.Player2hp > env.Player1hp else "TIE!!!")
            obs = env.reset()