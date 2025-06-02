import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import pygame
import map_utils
from hex import hexagon_map
from skins import tile
import skins
import minmax
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import obs_as_tensor
import time 
NUM_TILES = 31
NUM_PLACEMENTS = 19
NUM_ROTATIONS = 6

ACTION_SPACE_SIZE = NUM_PLACEMENTS * NUM_ROTATIONS * NUM_TILES


class NeuroHexEnv(gym.Env):
    def __init__(self):
        super(NeuroHexEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(3534) #each 114 actions mean-> xth tile into yth tile with zth rotation

        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(273,), dtype=np.float32) #273 = 21*13 (19 tiles + 2 hp info)
        self.choice = [[], []]
        self.screen = None
        self.map = None
        self.current_player = 0
        self.Player1hp, self.Player2hp = 20, 20
        self.done = False
        self.first_turn = True
        self.turn_started = False    
        self.all_player_tiles = skins.team_tiles[0] + [skins.teams_hq[0]]
            
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.choice = [[], []]
        self.screen = None
        self.map = hexagon_map(100, 50, 50)
        self.current_player = 0
        self.Player1hp, self.Player2hp = 20, 20
        self.done = False
        self.first_turn = True
        self.turn_started = False
        skins.team_tiles = skins.reset_tiles()
        self.all_player_tiles = skins.team_tiles[0] + [skins.teams_hq[0]]
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        reward = 0
        done = False
        chosen_tile = 0
        place_index = 0
        rotations = 0
        q=0
        r=0        
        tile_index, place_index, rotations = decode_action(action)
        for i, hex in enumerate(self.choice[0]):
            if hex.skin.equals(self.all_player_tiles[tile_index]):
                chosen_tile = i
                break
        
        battle=False
        selected_skin = self.choice[0][chosen_tile].skin
        if selected_skin.team == 3:
            battle=True
            self.Player1hp, self.Player2hp = map_utils.battle(self.map, self.Player1hp, self.Player2hp)
            self.choice[0].pop(chosen_tile)
        else:
            for i in range(rotations):
                selected_skin.rotate(1)

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
                                   
                    break 
        
        if len(self.choice[0])==1:
            self.choice[0].pop(0)
            self.current_player = 1
            self.turn_started = False 
        elif self.first_turn:
            self.current_player = 1
            self.turn_started = False    
                             
        map_value = 0
        enemy_attacked = 2
        enemy_hq_attacked = 6
        ally_hq_attacked = -8
        ally_attacked = -2        
        enemy_killed = 3
        ally_killed = -3
        placed_hex_attacked=-3
        if battle:
            enemy_attacked = 3
            ally_attacked = -3
            enemy_hq_attacked = 8
            ally_hq_attacked = -10

        attacked_hexes = []
        for hex in self.map.taken_hexes: #first check all tiles
            attacked_hexes.extend(map_utils.get_neighbors(self.map, hex.q, hex.r))
            attacked_hexes.extend(map_utils.get_neighbor_lines(self.map, hex.q, hex.r))
        
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
            attacked_hexes.extend(map_utils.get_neighbors(self.map, q, r))
            attacked_hexes.extend(map_utils.get_neighbor_lines(self.map, q, r))
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
            reward = 10 if self.Player1hp > self.Player2hp else -10
        #here False = done, additional done = True added at the end of the game
        return self._get_obs(), reward, False, {} #dict->info -> can return battle info


    def _get_obs(self):
        obs = np.zeros((19, 13), dtype=np.float32)  # 1 + 6 + 6

        index = 0
        for hex_row in self.map.hex_row_list:
            for hex_tile in hex_row.hex_list:
                # Column 0: team info
                if hex_tile.skin.team == 1:
                    obs[index, 0] = 1
                elif hex_tile.skin.team == 2:
                    obs[index, 0] = 2
                if hex_tile.skin.hq:
                    obs[index, 0] += 0.5

                # Columns 1–6: close attacks
                for direction in hex_tile.skin.close_attack:
                    if 0 <= direction <= 5:
                        obs[index, 1 + direction] = 1

                # Columns 7–12: ranged attacks
                for direction in hex_tile.skin.ranged_attack:
                    if 0 <= direction <= 5:
                        obs[index, 7 + direction] = 1

                index += 1

        # Normalize HPs
        player1_hp_normalized = self.Player1hp / 20.0
        player2_hp_normalized = self.Player2hp / 20.0

        hp_info = np.array([[player1_hp_normalized]*13, [player2_hp_normalized]*13], dtype=np.float32)

        full_obs = np.vstack([obs, hp_info])  # Shape (21, 13)
        return full_obs.flatten()  # Final shape: (273,)


    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((700, 700))
            self.font = pygame.font.Font(None, 36)
        self.screen.fill((0, 0, 0))
        map_utils.draw_map(self.screen, self.map)
        text1 = self.font.render(f"Player1 hp: {self.Player1hp}", True, (255, 255, 255))
        text2 = self.font.render(f"Player2 hp: {self.Player2hp}", True, (255, 255, 255))
        self.screen.blit(text1, (500, 200))
        self.screen.blit(text2, (500, 300))
        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()


    def _get_tile_type_index(self, tile):
        
        for i, template in enumerate(self.all_player_tiles):
            if tile.skin.equals(template):
                return i
        raise ValueError("Tile not found in template list")

    def get_action_mask(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        free_hexes = []
        index = 0
        for row in self.map.hex_row_list:
            for hex_tile in row.hex_list:
                if hex_tile.skin.team == 0:
                    free_hexes.append(index)
                index += 1

        active_tiles = self.choice[0]

        for i, tile in enumerate(active_tiles):
            tile_id = self._get_tile_type_index(tile)

            for placement in free_hexes:
                for rotation in range(NUM_ROTATIONS):
                    idx = encode_action(tile_id, placement, rotation)
                    mask[idx] = True

        return mask

def mask_fn(env):
    return env.get_action_mask()        


def masked_predict(model, obs, env, epsilon=0.1):
    if np.random.rand() < epsilon:
        valid_actions = np.where(env.get_action_mask())[0]
        return np.random.choice(valid_actions)

    obs_tensor = torch.tensor(obs).float().unsqueeze(0)
    q_values = model.policy.q_net(obs_tensor).detach().numpy()[0]
    mask = env.get_action_mask()
    masked_q_values = np.where(mask, q_values, -np.inf)
    action = np.argmax(masked_q_values)
    return action


def encode_action(tile_index, placement_index, rotation):

    return tile_index * NUM_PLACEMENTS * NUM_ROTATIONS + placement_index * NUM_ROTATIONS + rotation


def decode_action(action_id):
    tile_index = action_id // (NUM_PLACEMENTS * NUM_ROTATIONS)
    remainder = action_id % (NUM_PLACEMENTS * NUM_ROTATIONS)
    placement_index = remainder // NUM_ROTATIONS
    rotation = remainder % NUM_ROTATIONS
    return tile_index, placement_index, rotation    
     
     
if __name__ == "__main__":
    env = NeuroHexEnv()
    env = ActionMasker(env, mask_fn)
    
    policy_kwargs = dict(
        net_arch=[64, 64],
    )        
    model = MaskablePPO(#  try this on!!
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=32,
        n_steps=2048,
        learning_rate=1e-4,
        ent_coef=0.001,
        policy_kwargs=policy_kwargs
    )
    # model = MaskablePPO( #bigger model
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     batch_size=128,
    #     n_steps=4096,
    #     learning_rate=1e-4,
    #     ent_coef=0.001,  
    #     policy_kwargs=dict(net_arch=[128, 128])
    # )
    # model = MaskablePPO.load("neuroshima_ppo_model_400-000_minmax_vs_hard", env=env, device="cpu") #load model to learn
    results=[0,0,0]
    total_timesteps = 50_000
    obs = env.env.reset()[0]
    model._setup_learn(total_timesteps=total_timesteps*2)
    rollout_buffer = model.rollout_buffer
    rollout_buffer.reset()
    f = open("reward_time_ppo_minmax_100-000_vs_hard.csv", "w")
    tile_counter=0
    start_time = time.time()
    for step in range(total_timesteps*2):
        check_battle = True
        if rollout_buffer.full:            
            with torch.no_grad():
                next_obs_tensor = obs_as_tensor(obs, model.policy.device).unsqueeze(0)
                next_value = model.policy.predict_values(next_obs_tensor)

            rollout_buffer.compute_returns_and_advantage(
                last_values=next_value,
                dones=np.asarray([False], dtype=np.float32)
            )            
            model.train()
            rollout_buffer.reset()        
        if not env.env.turn_started:
            map_utils.fill_choice(env.env.choice, env.env.current_player, env.env.first_turn, False)
            env.env.turn_started = True
            
        if env.env.current_player == 0 and len(env.env.choice[0])>0:
            if len(env.env.map.free_hexes)==0:
                env.env.Player1hp, env.env.Player2hp = map_utils.battle(env.env.map, env.env.Player1hp, env.env.Player2hp)
            with torch.no_grad():
                obs_tensor = obs_as_tensor(obs, model.policy.device)
                obs_tensor = obs_tensor.unsqueeze(0)
                action_masks = get_action_masks(env)
                if not get_action_masks(env).any():
                    print("No valid actions for player")
                    if len(env.env.choice[0]) == 0:
                        env.env.current_player = 1
                        env.env.turn_started = False
                    elif len(env.env.map.free_hexes) == 0:
                        env.env.Player1hp, env.env.Player2hp = map_utils.battle(env.env.map, env.env.Player1hp, env.env.Player2hp)
                    else:
                        print("Choice not empty but no valid actions — check mask logic.")
                    continue             
                action, value, log_prob = model.policy.forward(obs_tensor, action_masks=action_masks)
                action = action.item()

            new_obs, reward, done, info = env.env.step(action)
            f.write(str(reward) + ",")
                                        
            rollout_buffer.add(obs, np.asarray(action), np.asarray(reward), done, value, log_prob, action_masks=action_masks)
            obs = new_obs

            if rollout_buffer.full:

                with torch.no_grad():
                    next_obs_tensor = obs_as_tensor(obs, model.policy.device).unsqueeze(0)
                    next_value = model.policy.predict_values(next_obs_tensor)

                rollout_buffer.compute_returns_and_advantage(
                    last_values=next_value,
                    dones=np.asarray([False], dtype=np.float32) #done = True only at the end of the game
                )                
                model.train()
                rollout_buffer.reset()


        if any(hex.skin.team == 0 for row in env.env.map.hex_row_list for hex in row.hex_list):
            check_battle = False

        if check_battle:
            env.env.Player1hp, env.env.Player2hp = map_utils.battle(env.env.map, env.env.Player1hp, env.env.Player2hp)                        

        if env.env.Player1hp <= 0 or env.env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
            final_reward = 0
            if env.env.Player1hp > env.env.Player2hp:
                final_reward = 10.0
                results[0]+=1
                f.write("\n"+"1won" + "\n")
                
            elif env.env.Player1hp < env.env.Player2hp:
                final_reward = -10.0
                results[1]+=1     
                f.write("\n"+"2won" + "\n")
                           
            else:
                results[2]+=1                
                final_reward = -1.0
                f.write("\n"+"tie" + "\n")
            with torch.no_grad():
                obs_tensor = obs_as_tensor(obs, model.policy.device).unsqueeze(0)
                dummy_mask = np.ones(env.action_space.n, dtype=bool)
                _, value, log_prob = model.policy.forward(obs_tensor, action_masks=dummy_mask)
            dones=np.asarray([True], dtype=np.float32)
            if not get_action_masks(env).any():
                print("no state to finish 1")
                dummy_mask = np.ones(env.action_space.n, dtype=bool)
                rollout_buffer.add(obs, np.asarray(0), np.asarray(final_reward), True, value, log_prob, action_masks=dummy_mask)                   
            else:      
                rollout_buffer.add(obs, np.asarray(0), np.asarray(final_reward), True, value, log_prob, action_masks=action_masks)  
        
            print("Player1 won" if env.env.Player1hp > env.env.Player2hp else "Player2 won" if env.env.Player2hp > env.env.Player1hp else "TIE!!!")
            obs = env.env.reset()[0]
            print(step)               
            continue
        
        if env.env.current_player == 1:
            map_utils.fill_choice(env.env.choice, env.env.current_player, env.env.first_turn, False)
            env.turn_started = True            
            if len(env.env.choice[1])>1 or env.env.first_turn:
                env.env.Player1hp, env.env.Player2hp = minmax.min_max(env.env.map, env.env.choice, 1, env.env.Player1hp, env.env.Player2hp, 0)  
            env.env.current_player = 0
            env.env.turn_started = False
            if env.env.first_turn:
                env.env.first_turn=False
                
        if any(hex.skin.team == 0 for row in env.env.map.hex_row_list for hex in row.hex_list):
            check_battle = False

        if check_battle:
            env.env.Player1hp, env.env.Player2hp = map_utils.battle(env.env.map, env.env.Player1hp, env.env.Player2hp)
        #env.render()
        if env.env.Player1hp <= 0 or env.env.Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
            final_reward = 0
            if env.env.Player1hp > env.env.Player2hp:
                final_reward = 10.0
                results[0]+=1
                f.write("\n"+"1won" + "\n")
                
            elif env.env.Player1hp < env.env.Player2hp:
                final_reward = -10.0
                results[1]+=1     
                f.write("\n"+"2won" + "\n")
                           
            else:
                results[2]+=1                
                final_reward = -1.0
                f.write("\n"+"tie" + "\n")
            
            with torch.no_grad():
                obs_tensor = obs_as_tensor(obs, model.policy.device).unsqueeze(0)
                dummy_mask = np.ones(env.action_space.n, dtype=bool)
                _, value, log_prob = model.policy.forward(obs_tensor, action_masks=dummy_mask)
            dones=np.asarray([True], dtype=np.float32)
            print(step)        
            if not get_action_masks(env).any():
                print("no state to finish 2")
                dummy_mask = np.ones(env.action_space.n, dtype=bool)
                rollout_buffer.add(obs, np.asarray(0), np.asarray(final_reward), True, value, log_prob, action_masks=dummy_mask)                   
            else:      
                rollout_buffer.add(obs, np.asarray(0), np.asarray(final_reward), True, value, log_prob, action_masks=dummy_mask)                   
            print("Player1 won" if env.env.Player1hp > env.env.Player2hp else "Player2 won" if env.env.Player2hp > env.env.Player1hp else "TIE!!!")
            obs = env.env.reset()[0]
    end_time = time.time()


    elapsed_time = end_time - start_time
    f.write("\n"+str(elapsed_time))
    f.close()
    env.env.reset()
    model.save("neuroshima_ppo_model_100-000_minmax_vs_hard")
    env.env.close()
    print("wins" +str(results[0]))
    print("losses" +str(results[1]))    
    print("ties" +str(results[2]))
    