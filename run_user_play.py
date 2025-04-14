from env import DinoEnv
import pygame

game = DinoEnv()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not game.game_over:
                game.step("jump")
            if event.key == pygame.K_SPACE and game.game_over:
                game.reset()

    if not game.game_over:
        game.step(None) 
    game.clock.tick(100)  

pygame.quit()

