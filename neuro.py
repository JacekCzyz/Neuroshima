import pygame
import numpy as np
from pygame.locals import *
from hex import hexagon_map
import map_utils
from skins import tile
import skins
import minmax

BUTTON_SIZE = 100
button1_rect = pygame.Rect(600, 600, BUTTON_SIZE, BUTTON_SIZE)  # Battle button
button2_rect = pygame.Rect(500, 600, BUTTON_SIZE, BUTTON_SIZE)  # Discard button

PlayerHP = np.array([20, 20])

pygame.init()
font = pygame.font.Font(None, 36)

size = np.array([700, 700])
screen = pygame.display.set_mode(size.tolist())
clock = pygame.time.Clock()

map = hexagon_map(100, 50, 50)
choice = np.array([[], []], dtype=object)
chosen_tile = 0
current_player = 0
turn_started = False
first_turn = True
check_battle = False
done = False
while not done:
    clock.tick(90)

    if not turn_started:
        choice = np.array(choice)
        choice = map_utils.fill_choice(choice.tolist(), current_player, first_turn, False)
        turn_started = True

    if current_player == 1:
        if len(choice[current_player]) > 1 or first_turn:
            PlayerHP[0], PlayerHP[1], choice= minmax.min_max(map, choice.tolist(), current_player, PlayerHP[0], PlayerHP[1], 1)
        current_player = 1 - current_player
        turn_started = False
        if first_turn:
            first_turn = False

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
            mouse_pos = np.array(pygame.mouse.get_pos())
            if button1_rect.collidepoint(mouse_pos.tolist()):
                PlayerHP[0], PlayerHP[1] = map_utils.battle(map, PlayerHP[0], PlayerHP[1])
                continue
            elif button2_rect.collidepoint(mouse_pos.tolist()):
                if chosen_tile < len(choice[current_player]):
                    choice[current_player] = np.delete(choice[current_player], chosen_tile)
                    current_player = 1 - current_player
                    turn_started = False
                continue

            check_battle = True
            if first_turn or len(choice[current_player]) > 1:
                selected_skin = choice[current_player][chosen_tile].skin
                if selected_skin.team == 3:
                    PlayerHP[0], PlayerHP[1] = map_utils.battle(map, PlayerHP[0], PlayerHP[1])
                    choice[current_player] = np.delete(choice[current_player], chosen_tile)
                else:
                    for i, hex in enumerate(map.free_hexes):
                        if hex.is_point_inside(mouse_pos.tolist()):
                            hex.skin = tile(
                                close_attack=selected_skin.close_attack.tolist(),
                                ranged_attack=selected_skin.ranged_attack.tolist(),
                                team=selected_skin.team,
                                initiative=selected_skin.initiative.tolist(),
                                lives=selected_skin.lives,
                                hq=selected_skin.hq,
                                color=selected_skin.color.tolist()
                            )
                            map.taken_hexes.append(hex)
                            map.free_hexes.pop(i)
                            choice[current_player] = np.delete(choice[current_player], chosen_tile)

                            if first_turn:
                                first_turn = current_player == 0
                                current_player = 1 - current_player
                                turn_started = False

                            if len(choice[current_player]) == 1:
                                choice[current_player] = np.delete(choice[current_player], 0)
                                current_player = 1 - current_player
                                turn_started = False
                            break


        if len(map.free_hexes)>0:
            check_battle=False


        if check_battle:
            PlayerHP[0], PlayerHP[1] = map_utils.battle(map, PlayerHP[0], PlayerHP[1])

    screen.fill((0, 0, 0))
    map_utils.draw_map(screen, map)

    for i, hex_tile in enumerate(choice[current_player]):
        map_utils.draw_hex(screen, hex_tile, chosen_tile == i)

    pygame.draw.rect(screen, (0, 255, 0), button1_rect)
    pygame.draw.rect(screen, (255, 0, 0), button2_rect)

    text1 = font.render(f"Player1 hp: {PlayerHP[0]}", True, (255, 255, 255))
    text2 = font.render(f"Player2 hp: {PlayerHP[1]}", True, (255, 255, 255))
    screen.blit(text1, (500, 200))
    screen.blit(text2, (500, 300))

    pygame.display.flip()

    if PlayerHP[0] <= 0 or PlayerHP[1] <= 0 or (len(skins.team_tiles[0]) <= 1 and len(skins.team_tiles[1]) <= 1):
        print("Player1 won" if PlayerHP[0] > PlayerHP[1] else "Player2 won" if PlayerHP[1] > PlayerHP[0] else "TIE!!!")
        done = True

pygame.quit()
