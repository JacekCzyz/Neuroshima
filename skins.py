import numpy as np

# attack directions: 0=mid-right, 1=top-right, 2=top-left, 3=mid-left, 4=bot-left, 5=bot-right
class tile:
    def __init__(self, close_attack=[], ranged_attack=[], team=None, initiative=[], lives=None, hq=False, color=(255, 255, 255)):
        self.close_attack = np.array(close_attack, dtype=int) if close_attack is not None else np.array([], dtype=int)
        self.ranged_attack = np.array(ranged_attack, dtype=int) if ranged_attack is not None else np.array([], dtype=int)
        self.initiative = np.array(initiative, dtype=int) if initiative is not None else np.array([], dtype=int)
        self.team = team
        self.lives = lives
        self.hq = hq
        self.color = np.array(color, dtype=int)
        
    def rotate(self, direction):
        self.close_attack = (self.close_attack + direction) % 6
        self.ranged_attack = (self.ranged_attack + direction) % 6

# Default skins and HQs
default_skin = tile()

team_1_hq = tile(team=1, lives=20, ranged_attack=[], initiative=[0], close_attack=[0,1,2,3,4,5], hq=True, color=[200, 0, 0])
team_2_hq = tile(team=2, lives=20, ranged_attack=[], initiative=[0], close_attack=[0,1,2,3,4,5], hq=True, color=[0, 200, 0])

teams_hq = {
    0: team_1_hq,
    1: team_2_hq
}

team1_color = [255, 0, 0]
team2_color = [0, 255, 0]

# Tiles for each team
team_tiles = np.array([
    [
        tile([], [], 1, [0], 0, False, team1_color) for _ in range(3)
    ] + [
        tile([], [], 1, [0], 3, False, team1_color) for _ in range(3)
    ] + [
        tile([], [3], 1, [3], 1, False, team1_color) for _ in range(3)
    ] + [
        tile([1, 1], [0], 1, [1], 2, False, team1_color) for _ in range(2)
    ] + [
        tile([1, 2, 3, 5], [], 1, [1], 1, False, team1_color) for _ in range(3)
    ] + [
        tile([], [1, 2, 3], 1, [1], 2, False, team1_color) for _ in range(2)
    ] + [
        tile([0, 1, 2, 3, 4, 5], [], 1, [2], 1, False, team1_color) for _ in range(3)
    ] + [
        tile([], [0, 2], 1, [2], 1, False, team1_color) for _ in range(2)
    ] + [
        tile([], [1, 2], 1, [2], 1, False, team1_color) for _ in range(2)
    ] + [
        tile([1, 2], [], 1, [2], 2, False, team1_color) for _ in range(2)
    ] + [
        tile([1, 1], [], 1, [2], 1, False, team1_color) for _ in range(2)
    ] + [
        tile([], [1], 1, [2, 1], 2, False, team1_color) for _ in range(2)
    ],
    [
        tile([], [], 2, [0], 0, False, team2_color) for _ in range(5)
    ] + [
        tile([3], [2], 2, [2], 1, False, team2_color) for _ in range(6)
    ] + [
        tile([1, 2, 2, 3], [], 2, [2], 1, False, team2_color) for _ in range(2)
    ] + [
        tile([1], [], 2, [3], 1, False, team2_color) for _ in range(5)
    ] + [
        tile([1, 1, 2, 2, 3, 3], [], 2, [1], 2, False, team2_color) for _ in range(2)
    ] + [
        tile([1, 2, 3], [], 2, [2], 2, False, team2_color) for _ in range(2)
    ] + [
        tile([2], [2], 2, [3], 1, False, team2_color) for _ in range(5)
    ] + [
        tile([2], [1, 3], 2, [2], 1, False, team2_color) for _ in range(4)
    ]
], dtype=object)