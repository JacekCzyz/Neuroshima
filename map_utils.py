import pygame
import math
import random
import numpy as np
import skins
from hex import hexagon

# Constants
choice_xses = [100, 200, 300]
AXIAL_OFFSETS = np.array([
    [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)],  # Row 0
    [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, 0), (1, 1)],  # Row 1
    [(0, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)],  # Row 2
    [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)],  # Row 3
    [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, 0), (1, 1)],
])

def get_axial_offset(direction, row):
    return AXIAL_OFFSETS[row, direction] if 0 <= row < AXIAL_OFFSETS.shape[0] else (0, 0)

def find_hex(hex_map, q, r):
    return next((hex for row in hex_map.hex_row_list for hex in row.hex_list if hex.q == q and hex.r == r), None)

def find_neighbor_in_line(hex_map, q, r, direction, row):
    neighbors = []
    main_hex = find_hex(hex_map, q, r)
    while True:
        if row>=0:
            dq, dr = get_axial_offset(direction, row)
            
            q += dq
            r += dr
            
            neighbor = find_hex(hex_map, q, r)

            if neighbor:
                if neighbor.skin.team != skins.default_skin.team and neighbor.skin.team != main_hex.skin.team:
                    neighbors.append(neighbor)
                    break
                row = neighbor.q                
            else:
                break
        else:
            break
    return neighbors

def find_neighbor(hex_map, q, r, direction, row):
    main_hex = find_hex(hex_map, q, r)
    if row>=0:
        dq, dr = get_axial_offset(direction, row)
        q += dq
        r += dr
            
        neighbor = find_hex(hex_map, q, r)
        if neighbor and neighbor.skin.team != skins.default_skin.team and neighbor.skin.team != main_hex.skin.team:
                return neighbor
    return None

def get_neighbor_lines(hex_map, q, r):
    neighbors = []
    hex = find_hex(hex_map, q, r)
    if len(hex.skin.ranged_attack)>0:
        for direction in hex.skin.ranged_attack:
            neighbors_in_line = find_neighbor_in_line(hex_map, q, r, direction, q)
            neighbors.extend(neighbors_in_line)
    return neighbors

def get_neighbors(hex_map, q, r):
    neighbors = []
    hex = find_hex(hex_map, q, r)
    if len(hex.skin.close_attack)>0:
        for direction in hex.skin.close_attack:
            neighbor_close = find_neighbor(hex_map, q, r, direction, q)
            if neighbor_close:
                neighbors.append(neighbor_close)
    return neighbors


def find_highest_init(hex_map):
    return max((max(hex.skin.initiative) for row in hex_map.hex_row_list for hex in row.hex_list if hex.skin.initiative.size > 0), default=0)

def draw_map(screen, hex_map):
    for row in hex_map.hex_row_list:
        for hex in row.hex_list:
            draw_hex(screen, hex, False)

def draw_hex(screen, hex, chosen):
    pygame.draw.polygon(screen, hex.skin.color.tolist(), hex.points)
    pygame.draw.polygon(screen, (0, 0, 255) if chosen else (0, 0, 0), hex.points, 2)

    center_x = sum(p[0] for p in hex.points) / 6
    center_y = sum(p[1] for p in hex.points) / 6

    font = pygame.font.Font(None, 24)
    row_spacing = 10  
    line_spacing = 20  

    if hex.skin.initiative.size > 0:
        for i, initiative in enumerate(hex.skin.initiative.tolist()):  
            text = font.render(str(initiative), True, (255, 255, 255))
            text_rect = text.get_rect(center=(center_x + i * row_spacing, center_y - 10))
            screen.blit(text, text_rect)

    text = font.render(str(hex.skin.lives), True, (255, 255, 255))
    text_rect = text.get_rect(center=(center_x, center_y - 10 + line_spacing))
    screen.blit(text, text_rect) 

    sqrt3_2 = math.sqrt(3) / 2
    side_centers = [
        ((hex.points[4][0] + hex.points[3][0]) / 2, (hex.points[4][1] + hex.points[3][1]) / 2),  # Mid-right (0)
        ((hex.points[5][0] + hex.points[4][0]) / 2, (hex.points[5][1] + hex.points[4][1]) / 2),  # Top-right (1)
        ((hex.points[5][0] + hex.points[0][0]) / 2, (hex.points[5][1] + hex.points[0][1]) / 2),  # Top-left (2)
        ((hex.points[0][0] + hex.points[1][0]) / 2, (hex.points[0][1] + hex.points[1][1]) / 2),  # Mid-left (3)
        ((hex.points[1][0] + hex.points[2][0]) / 2, (hex.points[1][1] + hex.points[2][1]) / 2),  # Bot-left (4)
        ((hex.points[2][0] + hex.points[3][0]) / 2, (hex.points[2][1] + hex.points[3][1]) / 2),  # Bot-right (5)
    ]
    side_normals = [
        (1, 0), (0.5, -sqrt3_2), (-0.5, -sqrt3_2),
        (-1, 0), (-0.5, sqrt3_2), (0.5, sqrt3_2)
    ]
    side_tangents = [
        (0, -1), (-sqrt3_2, -0.5), (-sqrt3_2, 0.5),
        (0, 1), (sqrt3_2, 0.5), (sqrt3_2, -0.5)
    ]

    offset_factor = 10
    spacing = 14
    lateral_shift = 6

    for direction in range(6):
        ranged_count = (hex.skin.ranged_attack == direction).sum()
        close_count = (hex.skin.close_attack == direction).sum()
        
        base_position = (
            side_centers[direction][0] - side_normals[direction][0] * offset_factor,
            side_centers[direction][1] - side_normals[direction][1] * offset_factor
        )
        normal, tangent = side_normals[direction], side_tangents[direction]

        for i in range(ranged_count):
            adjusted_spacing = spacing * (-0.5 - i * 0.5)
            offset_position = (
                base_position[0] + normal[0] * ((ranged_count - 1 - i) * adjusted_spacing) + tangent[0] * lateral_shift,
                base_position[1] + normal[1] * ((ranged_count - 1 - i) * adjusted_spacing) + tangent[1] * lateral_shift
            )
            draw_triangle(screen, offset_position)

        for i in range(close_count):
            adjusted_spacing = spacing * (-0.5 - i * 0.5)
            offset_position = (
                base_position[0] + normal[0] * ((close_count - 1 - i) * adjusted_spacing) - tangent[0] * lateral_shift,
                base_position[1] + normal[1] * ((close_count - 1 - i) * adjusted_spacing) - tangent[1] * lateral_shift
            )
            draw_dot(screen, offset_position)

def draw_triangle(screen, pos):
    size = 8
    x, y = pos
    height = math.sqrt(3) / 2 * size
    points = [(x, y - height / 2), (x - size / 2, y + height / 2), (x + size / 2, y + height / 2)]
    pygame.draw.polygon(screen, (0, 0, 0), points)

def draw_dot(screen, pos):
    pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 4)

def fill_choice(choice, current_player, first_turn, fake_fill):
    team_tiles = skins.team_tiles[current_player]

    # Ensure choice[current_player] is treated as a Python list
    choice[current_player] = list(choice[current_player])

    if first_turn:
        choice[current_player].append(hexagon(200, 500, 50, 10, 10, skins.teams_hq[current_player]))
    else:
        xs_needed = [xs for xs in choice_xses if all(hex.points[0][0] != xs for hex in choice[current_player])]
        source_tiles = team_tiles.tolist() if fake_fill else list(team_tiles)

        for xs in xs_needed:
            if not source_tiles:
                break
            idx = random.randint(0, len(source_tiles) - 1)
            tile = source_tiles[idx]
            choice[current_player].append(hexagon(xs, 500, 50, 10, 10, tile))

            if not fake_fill:
                skins.team_tiles[current_player] = np.delete(team_tiles, idx)
            else:
                del source_tiles[idx]

    # Convert back to NumPy object array if necessary
    choice[current_player] = np.array(choice[current_player], dtype=object)
    if current_player==0:
        choice = np.array([choice[current_player], []], dtype=object)
    else:
        choice = np.array([[], choice[current_player]], dtype=object)        
    return choice

# def battle(hex_map, hp1, hp2):
#     for i in range(find_highest_init(hex_map), -1, -1):
#         for hex in hex_map.taken_hexes:
#             if i not in hex.skin.initiative:
#                 continue
#             for target in get_neighbors(hex_map, hex.q, hex.r) + get_neighbor_lines(hex_map, hex.q, hex.r):
#                 if target is None:
#                     if target.skin.lives is not None:
#                         target.skin.lives -= 1
#                         if target.skin.hq:
#                             if target.skin.team == 1:
#                                 hp1 -= 1
#                             else:
#                                 hp2 -= 1
#                         if target.skin.lives == 0:
#                             hex_map.taken_hexes.remove(target)
#                             hex_map.free_hexes.append(target)
#                             target.skin = skins.default_skin
#     return hp1, hp2

def battle(hex_map, player1hp, player2hp):
    init = find_highest_init(hex_map)
    for i in range(init, -1, -1):
        attacked_hexes = []
        for hex in hex_map.taken_hexes:
            if i in hex.skin.initiative:
                attacked_hexes.extend(get_neighbors(hex_map, hex.q, hex.r))
                attacked_hexes.extend(get_neighbor_lines(hex_map, hex.q, hex.r))
        
        for attacked_hex in attacked_hexes:
            if attacked_hex.skin.lives is not None:
                attacked_hex.skin.lives -= 1
                if attacked_hex.skin.hq:
                    if attacked_hex.skin.team == 1:
                        player1hp -= 1
                    else:
                        player2hp -= 1
                if attacked_hex.skin.lives == 0:
                    hex_map.taken_hexes.remove(attacked_hex)
                    hex_map.free_hexes.append(attacked_hex)
                    attacked_hex.skin = skins.default_skin
    return player1hp, player2hp