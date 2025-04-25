import gym
from gym import spaces
import numpy as np
import pygame
import map_utils
from hex import hexagon_map
from skins import tile
import skins

class NeuroHexEnv(gym.Env):
    def __init__(self):
        super(NeuroHexEnv, self).__init__()
        
        self.action_space = spaces.Discrete(342)

        self.observation_space = spaces.Box(low=0, high=255, shape=(100,), dtype=np.float32)

        self.screen = None
        self.map = None
        self.current_player = 0
        self.Player1hp, self.Player2hp = 20, 20
        self.done = False
        self.reset()

    def reset(self):
        self.map = hexagon_map(100, 50, 50)
        self.Player1hp, self.Player2hp = 20, 20
        self.current_player = 0
        self.done = False
        
        return self._get_obs()

    def step(self, action):
        reward = 0
        done = False

        if action == 0:
            pass
        elif action == 1:
            pass
        elif action == 2:
            pass
        elif action == 3:
            pass

        if self.Player1hp <= 0 or self.Player2hp <= 0:
            done = True
            reward = 1 if self.Player1hp > self.Player2hp else -1

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.zeros(100, dtype=np.float32)  # temp 

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((700, 700))
        self.screen.fill((0, 0, 0))
        map_utils.draw_map(self.screen, self.map)
        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()



if __name__ == "__main__":
    env = NeuroHexEnv(render_mode=True)

    for _ in range(1000):
        action = env.action_space.sample()
        
        # Wykonanie akcji w środowisku
        state, reward, done, info = env.step(action)
        
        # Renderowanie aktualnego stanu gry
        env.render()
        rewardMain += reward

        # Jeśli gra się skończyła, wypisanie "Game Over" i zakończenie
        if done:
            print("Game Over")
            break
    
    print(rewardMain)
    
    env.close()  # Zamknięcie środowiska