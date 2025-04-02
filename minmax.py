import map_utils
import skins
from skins import tile
import hex
from hex import hexagon_map
import copy

class MoveValue:
    def __init__(self, choice_indexes, qs, rs, rotations, free_index, value):
        self.choice_indexes = choice_indexes
        self.qs = qs
        self.rs = rs
        self.rotations = rotations
        self.free_index = free_index
        self.value = value

def calculate_map_value(hex_map, player, battle):
    map_value=0
    ally_team = 1
    if player == 1:
        ally_team=2
    
    enemy_attacked = 2
    ally_attacked = -2
    enemy_hq_attacked = 4
    ally_hq_attacked = -5
    enemy_killed = 2
    ally_killed = -2
    if battle:
        enemy_attacked = 3
        ally_attacked = -3
        enemy_hq_attacked = 5
        ally_hq_attacked = -6

    attacked_hexes = []
    for hex in hex_map.taken_hexes:
        attacked_hexes.extend(map_utils.get_neighbors(hex_map, hex.q, hex.r))
        attacked_hexes.extend(map_utils.get_neighbor_lines(hex_map, hex.q, hex.r))
       
    checked_state = []
    for hex in attacked_hexes:
        if hex.skin.team==ally_team:
            if hex.skin.hq:
                map_value+=ally_hq_attacked
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
                
    return map_value

def min_max(hex_map, choice, current_player, player1hp, player2hp, depth):
    best_move = perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth, float('-inf'), float('inf'))
    return make_final_move(hex_map, best_move, choice, current_player, player1hp, player2hp)

def perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth, alpha, beta):
    best_value = float('-inf') if current_player == 1 else float('inf')
    best_move = None
    temp_choice = copy.deepcopy(choice)  
    if len(temp_choice[current_player])==0:
        map_utils.fill_choice(temp_choice, current_player, False, True)
    elif len(temp_choice[current_player])<3 and not temp_choice[current_player][0].skin.hq:
        map_utils.fill_choice(temp_choice, current_player, False, True) 
        
    if depth == 0:
        possible_moves = find_cur_values(hex_map, current_player, temp_choice, player1hp, player2hp)
        for move in possible_moves:
            if (current_player == 0 and move.value < best_value) or (current_player == 1 and move.value > best_value):
                best_value = move.value
                best_move = move
        return best_move if best_move else possible_moves[0]

    possible_moves = find_cur_values(hex_map, current_player, choice, player1hp, player2hp)

    next_player = 1 - current_player
    for move in sorted(possible_moves, key=lambda mv: -mv.value if current_player == 1 else mv.value):
        next_map = hex_map.set_mockup_map()
        move_value = perform_min_max(next_map, choice, next_player, player1hp, player2hp, depth - 1, alpha, beta)
        
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


import copy

def find_cur_values(hex_map, player, choice, player1hp, player2hp):
    first_moves = []
    second_moves = []
    
    choice_player = choice[player]
    choice_len = len(choice_player)
    
    initial_map = hex_map.set_mockup_map()
    for i, ihex in enumerate(choice_player):
        first_map = copy.copy(initial_map)

        if ihex.skin.team == 3:
            player1hp, player2hp = map_utils.battle(first_map, player1hp, player2hp)
            first_moves.append(MoveValue([i], [1110], [1110], [1110], [1110], calculate_map_value(first_map, player, True)))
        else:
            for j in range(len(hex_map.free_hexes)):
                first_map = copy.copy(first_map)
                
                ihex_move = calc_moves(first_map, ihex, j, player)
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
                    temp_skin = first_map.free_hexes[move.free_index[0]].skin
                    first_map.free_hexes[move.free_index[0]].skin = tile(
                        hex.skin.color, hex.skin.close_attack, hex.skin.ranged_attack,
                        hex.skin.team, hex.skin.initiative, hex.skin.lives, hex.skin.hq
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

                if hex.skin.team != 3:
                    first_map.free_hexes.insert(move.free_index[0], first_map.taken_hexes.pop())
                    first_map.free_hexes[move.free_index[0]].skin = temp_skin  

        return second_moves
    else:
        return first_moves


def calc_moves(hex_map, hex, free_index, player):
    best_value = float('-inf') if player == 1 else float('inf')
    best_move=None
    for i in range(6):
        hex_map.free_hexes[free_index].skin = tile(hex.skin.color, hex.skin.close_attack, hex.skin.ranged_attack,
                                hex.skin.team, hex.skin.initiative, hex.skin.lives, hex.skin.hq)
        hex_map.taken_hexes.append(hex_map.free_hexes[free_index])
        hex_map.free_hexes.pop(free_index)
        
        move = (hex_map.taken_hexes[-1].q, hex_map.taken_hexes[-1].r, i, calculate_map_value(hex_map, player, False))
        if (player==1 and move[3]>best_value) or (player==0 and move[3]<best_value):
            best_value = move[3]
            best_move = move
            
        hex_map.taken_hexes[-1].skin = skins.default_skin
        hex_map.free_hexes.insert(free_index,hex_map.taken_hexes[-1])        
        hex_map.taken_hexes.pop(-1)
        
        hex.skin.rotate(1)
        
    return best_move


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
                    hex.skin = tile(selected_skin.color, selected_skin.close_attack, selected_skin.ranged_attack,
                    selected_skin.team, selected_skin.initiative, selected_skin.lives, selected_skin.hq)
                    hex_map.taken_hexes.append(hex)
                    hex_map.free_hexes.pop(j)                            
                    break
    if len(move_val.choice_indexes)>1:
        choice[player].clear()
    else:
        choice[player].pop(max(move_val.choice_indexes))

    return player1hp, player2hp