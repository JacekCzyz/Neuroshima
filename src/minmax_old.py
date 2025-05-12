import map_utils
import skins
from skins import tile
import hex
from hex import hexagon_map

class MoveValue:
    def __init__(self, choice_indexes, qs, rs, rotations, free_index, value):
        self.choice_indexes = choice_indexes
        self.qs = qs
        self.rs = rs
        self.rotations = rotations
        self.free_index = rotations
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
    
    best_move = perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth)
    
    return make_final_move(hex_map, best_move, choice, current_player, player1hp, player2hp)

    
def perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth):
    if depth == 0:
        possible_moves = find_cur_values(hex_map, current_player, choice, player1hp, player2hp)

        for move in possible_moves:
            next_map = hex_map.set_mockup_map()
            move_value = perform_min_max(next_map, choice, next_player, player1hp, player2hp, depth - 1)

            if move_value:  # Ensure move_value is not None before comparison
                if (current_player == 0 and move_value[2].value > best_value) or (current_player == 1 and move_value[2].value < best_value):
                    best_value = move_value[2].value
                    best_move = move

        return best_move if best_move else possible_moves[0]
    
    best_value = float('-inf') if current_player == 0 else float('inf')
    best_move = None

    possible_moves = find_cur_values(hex_map, current_player, choice, player1hp, player2hp)
    if not possible_moves:
        return None  # No valid moves

    next_player = 1 - current_player

    for move in possible_moves:
        next_map = hex_map.set_mockup_map()
        move_value = perform_min_max(next_map, choice, next_player, player1hp, player2hp, depth - 1)

        if move_value:  # Ensure move_value is not None before comparison
            if (current_player == 0 and move_value[2].value > best_value) or (current_player == 1 and move_value[2].value < best_value):
                best_value = move_value[2].value
                best_move = move

    return best_move if best_move else possible_moves[0]  # Fallback to the first move if none is better


# def find_cur_values(hex_map, player, choice, player1hp, player2hp):
#     values = []
#     for i, ihex in enumerate(choice[player]):
#         for dir in range(6):
#             start_map = hex_map.set_mockup_map()
#             moves = make_calc_move(start_map, ihex, player, player1hp, player2hp)
            
#             for qi, ri, value in moves:
#                 values.append(MoveValue([i], [qi], [ri], [dir], value))
            
#             ihex.skin.rotate(1)
#         ihex.skin.rotate(1)
#     return values

# def make_calc_move(hex_map, hex, player, player1hp, player2hp):
#     possible_moves = []

#     if hex.skin.team == 3:
#         player1hp, player2hp = map_utils.battle(hex_map, player1hp, player2hp)
#         possible_moves.append((1110, 1110, calculate_map_value(hex_map, player, True)))
#     else:
#         for placed_hex in hex_map.free_hexes:
#             placed_hex.skin = tile(hex.skin.color, hex.skin.close_attack, hex.skin.ranged_attack,
#                                    hex.skin.team, hex.skin.initiative, hex.skin.lives, hex.skin.hq)
#             hex_map.taken_hexes.append(placed_hex)
#             hex_map.free_hexes.remove(placed_hex)
          
#             move_value = calculate_map_value(hex_map, player, False)
#             possible_moves.append((placed_hex.q, placed_hex.r, move_value))
            
#             placed_hex.skin = skins.default_skin
#             hex_map.taken_hexes.remove(placed_hex)
    
#     return possible_moves

def make_calc_move(hex_map, hex, player, player1hp, player2hp):
    moves=[]
    if hex.skin.team == 3:
        player1hp, player2hp = map_utils.battle(hex_map, player1hp, player2hp)
        moves.append((1110, 1110, calculate_map_value(hex_map, player, True)))
    else:
        for i in range(len(hex_map.free_hexes)):
            hex_map.free_hexes[i].skin = tile(hex.skin.color, hex.skin.close_attack, hex.skin.ranged_attack,
                                hex.skin.team, hex.skin.initiative, hex.skin.lives, hex.skin.hq)
            hex_map.taken_hexes.append(hex_map.free_hexes[i])
            hex_map.free_hexes.pop(i)
        
            moves.append((hex_map.taken_hexes[-1].q, hex_map.taken_hexes[-1].r, calculate_map_value(hex_map, player, False)))
        
            hex_map.taken_hexes[-1].skin = skins.default_skin
            hex_map.free_hexes.insert(i,hex_map.taken_hexes[-1])        
            hex_map.taken_hexes.pop(-1)
    return moves

# def perform_min_max(hex_map, choice, current_player, player1hp, player2hp, depth):
#     values = [find_cur_value(hex_map, current_player, current_player, choice, player1hp, player2hp) for _ in range(depth)]
#     return make_final_move(hex_map, values[0], choice, current_player, player1hp, player2hp)

def find_cur_values(hex_map, player, choice, player1hp, player2hp):
    values = []
    for i in range(len(choice[player])):
        stable_map = hex_map
        for dir in range(6):
            start_map = stable_map.set_mockup_map()
            ihex = choice[player][i]
            first_moves_list = make_calc_move(start_map, ihex, player, player1hp, player2hp)
            temp_map = start_map.set_mockup_map()
            
            for move in first_moves_list:
                (qi,ri,value1) = move
                
                hex_to_change = None
                k=0
                for l in range(len(temp_map.free_hexes)):
                    if temp_map.free_hexes[l].q == qi and temp_map.free_hexes[l].r == ri:
                        k=l
                        hex_to_change = temp_map.free_hexes[l]
                
                if hex_to_change!=None:
                    hex_to_change.skin = tile(ihex.skin.color, ihex.skin.close_attack, ihex.skin.ranged_attack,
                                                        ihex.skin.team, ihex.skin.initiative, ihex.skin.lives, ihex.skin.hq)
                    temp_map.taken_hexes.append(hex_to_change)
                    temp_map.free_hexes.pop(k) 
                if len(choice[player])>1:
                    for j in range(len(choice[player])):
                        if i==j:
                            continue
                        jhex = choice[player][j]
                        for dir2 in range(6):
                            second_moves = make_calc_move(temp_map, jhex, player, player1hp, player2hp)
                            for second_move in second_moves:
                                (qj, rj, value2) = second_move
                                values.append(MoveValue([i,j], [qi,qj], [ri,rj], [dir,dir2], value2)) 
                            jhex.skin.rotate(1)
                        jhex.skin.rotate(1)
                else:
                    values.append(MoveValue([i], [qi], [ri], [dir], value1))
                        
                if hex_to_change!=None:
                    hex_to_change.skin = skins.default_skin
                    temp_map.free_hexes.insert(k, hex_to_change)
                    temp_map.taken_hexes.pop(-1)
            ihex.skin.rotate(1)
        ihex.skin.rotate(1) #one extra for starting position

    return values        
    # min_value = values[0]
    # max_value = values[0]
    # for value in values:
    #     if value.value<min_value:
    #         min_value=value
    #     elif value.value>max_value:
    #         max_value=value 
            
    # if starting_player==player: #in order to simplify, only return the best option
    #     return max_value
    # else:
    #     return min_value

# def make_calc_move(hex_map, hex, player, player1hp, player2hp):
#     if hex.skin.team == 3:
#         player1hp, player2hp = map_utils.battle(hex_map, player1hp, player2hp)
#         return (1110, 1110, calculate_map_value(hex_map, player, True))
    
#     best_move = None
#     for i in range(len(hex_map.free_hexes)):
#         hex_map.free_hexes[i].skin = tile(hex.skin.color, hex.skin.close_attack, hex.skin.ranged_attack,
#                                hex.skin.team, hex.skin.initiative, hex.skin.lives, hex.skin.hq)
#         hex_map.taken_hexes.append(hex_map.free_hexes[i])
#         hex_map.free_hexes.pop(i)
      
#         move_value = calculate_map_value(hex_map, player, False)
#         if not best_move or move_value > best_move[2]:
#             best_move = (hex_map.taken_hexes[-1].q, hex_map.taken_hexes[-1].r, move_value)
      
#         hex_map.taken_hexes[-1].skin = skins.default_skin
#         hex_map.free_hexes.insert(i,hex_map.taken_hexes[-1])        
#         hex_map.taken_hexes.pop(-1)
    
#     if best_move:
#         return best_move
#     else:
#         return (-1, -1, calculate_map_value(hex_map, player, False))

def make_final_move(hex_map, move_val, choice, player, player1hp, player2hp):
    for i in range(len(move_val.choice_indexes)):
        for j in range(len(move_val.rotations)):
            choice[player][move_val.choice_indexes[i]].skin.rotate(1)
            
        if choice[player][move_val.choice_indexes[i]].skin.team == 3:
            player1hp, player2hp = map_utils.battle(hex_map, player1hp, player2hp)
        else:              
            for j in range(len(hex_map.free_hexes)):
                if hex_map.free_hexes[j].q == move_val.qs[i] and hex_map.free_hexes[j].r == move_val.rs[i]:
                    selected_skin = choice[player][move_val.choice_indexes[i]].skin
                    hex = hex_map.free_hexes[j]
                    hex.skin = tile(selected_skin.color, selected_skin.close_attack, selected_skin.ranged_attack,
                    selected_skin.team, selected_skin.initiative, selected_skin.lives, selected_skin.hq)
                    hex_map.taken_hexes.append(hex)
                    hex_map.free_hexes.pop(j)                            
                    break
    choice[player].pop(max(move_val.choice_indexes))
    if len(move_val.choice_indexes)>1:
        choice[player].pop(min(move_val.choice_indexes))
    
    return player1hp, player2hp