import map_utils
import skins
from skins import tile
import hex
from hex import hexagon_map
import copy
import numpy as np
class MoveValue:
    def __init__(self, choice_indexes, qs, rs, rotations, free_index, value):
        self.choice_indexes = np.array(choice_indexes)
        self.qs = np.array(qs)
        self.rs = np.array(rs)
        self.rotations = np.array(rotations)
        self.free_index = np.array(free_index)
        self.value = value

def calculate_map_value(hex_map, player, battle):
    map_value = 0
    ally_team = 2 if player == 1 else 1
    
    enemy_attacked, ally_attacked = (3, -3) if battle else (2, -2)
    enemy_hq_attacked, ally_hq_attacked = (5, -6) if battle else (4, -5)
    enemy_killed, ally_killed = 2, -2
    
    attacked_hexes = np.array([])
    for hex in hex_map.taken_hexes:
        attacked_hexes = np.append(attacked_hexes, map_utils.get_neighbors(hex_map, hex.q, hex.r))
        attacked_hexes = np.append(attacked_hexes, map_utils.get_neighbor_lines(hex_map, hex.q, hex.r))
    
    checked_state = np.array([])
    for hex in attacked_hexes:
        if hex.skin.team == ally_team:
            map_value += ally_hq_attacked if hex.skin.hq else ally_attacked
            if np.count_nonzero(attacked_hexes == hex) >= hex.skin.lives and hex not in checked_state:
                map_value += ally_killed
                checked_state = np.append(checked_state, hex)
        else:
            map_value += enemy_hq_attacked if hex.skin.hq else enemy_attacked
            if np.count_nonzero(attacked_hexes == hex) >= hex.skin.lives and hex not in checked_state:
                map_value += enemy_killed
                checked_state = np.append(checked_state, hex)
    
    return map_value

def min_max(hex_map, choice, current_player, player1hp, player2hp, depth):
    best_move = perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth, float('-inf'), float('inf'))
    return make_final_move(hex_map, best_move, choice, current_player, player1hp, player2hp)

def perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth, alpha, beta):
    best_value = float('-inf') if current_player == 1 else float('inf')
    best_move = None
    temp_choice = copy.deepcopy(choice)
    
    if len(temp_choice[current_player]) == 0 or (len(temp_choice[current_player]) < 3 and not temp_choice[current_player][0].skin.hq):
        temp_choice = map_utils.fill_choice(temp_choice, current_player, False, True)
    
    if depth == 0:
        possible_moves = find_cur_values(hex_map, current_player, temp_choice, player1hp, player2hp)
        best_move = min(possible_moves, key=lambda mv: mv.value) if current_player == 0 else max(possible_moves, key=lambda mv: mv.value)
        return best_move if best_move else possible_moves[0]
    
    possible_moves = find_cur_values(hex_map, current_player, temp_choice, player1hp, player2hp)
    next_player = 1 - current_player
    
    for move in sorted(possible_moves, key=lambda mv: -mv.value if current_player == 1 else mv.value):
        next_map = hex_map.set_mockup_map()
        move_value = perform_min_max(next_map, temp_choice, next_player, player1hp, player2hp, depth - 1, alpha, beta)
        
        if move_value:
            if current_player == 1:
                if move_value.value > best_value:
                    best_value = move_value.value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if move_value.value < best_value:
                    best_value = move_value.value
                    best_move = move
                beta = min(beta, best_value)
        
        if beta <= alpha:
            break
    
    return best_move if best_move else (possible_moves[0] if possible_moves else None)


def find_cur_values(hex_map, player, choice, player1hp, player2hp):
    first_moves = []
    second_moves = []
    
    choice_player = np.array(choice[player])
    choice_len = len(choice_player)
    
    initial_map = hex_map.set_mockup_map()
    for i, ihex in enumerate(choice_player):
        first_map = copy.copy(initial_map)

        if ihex.skin.team == 3:
            player1hp, player2hp = map_utils.battle(first_map, player1hp, player2hp)
            first_moves.append(MoveValue([i], [1110], [1110], [1110], [1110], calculate_map_value(first_map, player, True)))
        else:
            for j in range(len(hex_map.free_hexes)):
                first_map2 = copy.copy(first_map)
                
                ihex_move = calc_moves(first_map2, ihex, j, player)
                if ihex_move:
                    first_moves.append(MoveValue([i], [ihex_move[0]], [ihex_move[1]], [ihex_move[2]], [j], ihex_move[3]))

    
    if choice_len == 3:
        for k, khex in enumerate(choice_player):
            for move in first_moves:
                if k == move.choice_indexes[0]:
                    continue

                first_map = initial_map.set_mockup_map()
                hex = choice_player[move.choice_indexes[0]]

                if hex.skin.team == 3:
                    player1hp, player2hp = map_utils.battle(first_map, player1hp, player2hp)
                else:
                    if move.choice_indexes[0] < k:
                        temp_skin = first_map.free_hexes[move.free_index[0]].skin
                        first_map.free_hexes[move.free_index[0]].skin = tile(
                            color=hex.skin.color, close_attack=hex.skin.close_attack, ranged_attack=hex.skin.ranged_attack,
                            team=hex.skin.team, initiative=hex.skin.initiative, lives=hex.skin.lives, hq=hex.skin.hq
                        )
                        first_map.taken_hexes.append(first_map.free_hexes.pop(move.free_index[0]))

                if khex.skin.team == 3:
                    second_map = copy.copy(first_map)
                    player1hp, player2hp = map_utils.battle(second_map, player1hp, player2hp)
                    second_moves.append(MoveValue(
                        [move.choice_indexes[0], k], [move.qs[0], 1110],
                        [move.rs[0], 1110], [move.rotations[0], 1110], 
                        [move.free_index[0], 1110], calculate_map_value(second_map, player, True)
                    ))
                else:                
                    for l in range(len(first_map.free_hexes)):
                        second_map = copy.copy(first_map)
                        khex_move = calc_moves(second_map, khex, l, player)
                        if khex_move:
                            second_moves.append(MoveValue([move.choice_indexes[0],k], [move.qs[0],khex_move[0]], [move.rs[0],khex_move[1]], [move.rotations[0],khex_move[2]], [move.free_index[0], l], khex_move[3]))
                if hex.skin.team != 3 and move.choice_indexes[0] < k:
                    first_map.free_hexes.insert(move.free_index[0], first_map.taken_hexes.pop())
                    first_map.free_hexes[move.free_index[0]].skin = temp_skin  

        return np.array(second_moves)
    else:     
        return np.array(first_moves)


def calc_moves(hex_map, hex, free_index, player):
    best_value = float('-inf') if player == 1 else float('inf')
    best_move = None
    for i in range(6):
        hex_map.free_hexes[free_index].skin = tile(
            color=np.array(hex.skin.color),
            close_attack=np.array(hex.skin.close_attack),
            ranged_attack=np.array(hex.skin.ranged_attack),
            team=hex.skin.team,
            initiative=np.array(hex.skin.initiative),
            lives=np.array(hex.skin.lives),
            hq=hex.skin.hq
        )

        hex_map.taken_hexes.append(hex_map.free_hexes[free_index])
        hex_map.free_hexes.pop(free_index)
        
        move = (hex_map.taken_hexes[-1].q, hex_map.taken_hexes[-1].r, i, calculate_map_value(hex_map, player, False))
        if (player == 1 and move[3] > best_value) or (player == 0 and move[3] < best_value):
            best_value = move[3]
            best_move = move

            
        hex_map.taken_hexes[-1].skin = skins.default_skin
        hex_map.free_hexes.insert(free_index, hex_map.taken_hexes.pop())        
        hex.skin.rotate(1)
        
    return best_move


import numpy as np

def make_final_move(hex_map, move_val, choice, player, player1hp, player2hp):
    for i in range(len(move_val.choice_indexes)):
        if choice[player][move_val.choice_indexes[i]].skin.team == 3:
            player1hp, player2hp = map_utils.battle(hex_map, player1hp, player2hp)
        else:
            for x in range(move_val.rotations[i]):
                choice[player][move_val.choice_indexes[i]].skin.rotate(1)
            for j in range(len(hex_map.free_hexes)):
                if hex_map.free_hexes[j].q == move_val.qs[i] and hex_map.free_hexes[j].r == move_val.rs[i]:
                    selected_skin = choice[player][move_val.choice_indexes[i]].skin
                    hex = hex_map.free_hexes[j]
                    hex.skin = tile(
                        color=selected_skin.color,
                        close_attack=selected_skin.close_attack,
                        ranged_attack=selected_skin.ranged_attack,
                        team=selected_skin.team,
                        initiative=selected_skin.initiative,
                        lives=selected_skin.lives,
                        hq=selected_skin.hq,
                    )
                    hex_map.taken_hexes.append(hex)
                    hex_map.free_hexes.pop(j)
                    break

    # Convert choice[player] back to a numpy array
    choice[player] = np.array([], dtype=object)

    return player1hp, player2hp, choice
