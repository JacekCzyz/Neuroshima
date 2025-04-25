import pygame
from pygame.locals import *
from hex import hexagon_map
import map_utils
from skins import tile
import skins
import minmax

BUTTON_SIZE = 100
button1_rect = pygame.Rect(600, 600, BUTTON_SIZE, BUTTON_SIZE)  # Battle button
button2_rect = pygame.Rect(500, 600, BUTTON_SIZE, BUTTON_SIZE)  # Discard button

Player1hp, Player2hp = 20, 20

pygame.init()
font = pygame.font.Font(None, 36)

size = [700, 700]
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

map = hexagon_map(100, 50, 50)
choice = [[], []]
chosen_tile = 0
current_player = 0
turn_started = False
first_turn = True

done = False
while not done:
    clock.tick(90)

    if not turn_started:
        map_utils.fill_choice(choice, current_player, first_turn, False)
        turn_started = True

    if current_player==1:
        if len(choice[current_player])>1 or first_turn:
            Player1hp, Player2hp = minmax.min_max(map, choice, current_player, Player1hp, Player2hp, 1)  
        current_player = 1 - current_player
        turn_started = False
        if first_turn:
            first_turn=False      

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            continue
        if event.type == pygame.KEYDOWN:
            key = event.key
            if pygame.K_1 <= key <= pygame.K_3:
                chosen_tile = key - pygame.K_1
            elif key == pygame.K_LEFT:
                choice[current_player][chosen_tile].skin.rotate(-1)
            elif key == pygame.K_RIGHT:
                choice[current_player][chosen_tile].skin.rotate(1)
            continue
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if button1_rect.collidepoint(mouse_pos):
                Player1hp, Player2hp = map_utils.battle(map, Player1hp, Player2hp)
                continue
            elif button2_rect.collidepoint(mouse_pos):
                if chosen_tile < len(choice[current_player]):
                    choice[current_player].pop(chosen_tile)
                    current_player = 1 - current_player
                    turn_started = False
                continue

            check_battle = True
            if first_turn or len(choice[current_player])>1:
                selected_skin = choice[current_player][chosen_tile].skin
                if selected_skin.team == 3:
                    Player1hp, Player2hp = map_utils.battle(map, Player1hp, Player2hp)
                    choice[current_player].pop(chosen_tile)
                else:
                    for i in range(len(map.free_hexes)):
                        hex = map.free_hexes[i]
                        if hex.is_point_inside(mouse_pos):
                            hex.skin = tile(selected_skin.color, selected_skin.close_attack, selected_skin.ranged_attack,
                                                selected_skin.team, selected_skin.initiative, selected_skin.lives, selected_skin.hq)
                            map.taken_hexes.append(hex)
                            map.free_hexes.pop(i)                            
                            choice[current_player].pop(chosen_tile)

                            if first_turn:
                                first_turn = current_player == 0
                                current_player = 1 - current_player
                                turn_started = False
                                    
                            if len(choice[current_player])==1:
                                choice[current_player].pop(0)
                                current_player = 1 - current_player
                                turn_started = False  
                            break                        
                      
        if any(hex.skin.team == 0 for row in map.hex_row_list for hex in row.hex_list):
            check_battle = False

        if check_battle:
            Player1hp, Player2hp = map_utils.battle(map, Player1hp, Player2hp)
                

    screen.fill((0, 0, 0))
    map_utils.draw_map(screen, map)

    for i, hex_tile in enumerate(choice[current_player]):
        map_utils.draw_hex(screen, hex_tile, chosen_tile == i)

    pygame.draw.rect(screen, (0, 255, 0), button1_rect)
    pygame.draw.rect(screen, (255, 0, 0), button2_rect)

    text1 = font.render(f"Player1 hp: {Player1hp}", True, (255, 255, 255))
    text2 = font.render(f"Player2 hp: {Player2hp}", True, (255, 255, 255))
    screen.blit(text1, (500, 200))
    screen.blit(text2, (500, 300))

    pygame.display.flip()

    if Player1hp <= 0 or Player2hp <= 0 or (len(skins.team_tiles[0])<=1 and len(skins.team_tiles[1])<=1):
        print("Player1 won" if Player1hp > Player2hp else "Player2 won" if Player2hp > Player1hp else "TIE!!!")
        done = True

pygame.quit()