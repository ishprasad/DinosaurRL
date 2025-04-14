import pygame
import random

pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class DinoEnv:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Dino Game")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.dino_y = SCREEN_HEIGHT - 60 
        self.dino_velocity = 0
        self.gravity = 0.8
        self.jump_power = -15
        self.obstacles = []
        self.obstacle_timer = 0
        self.score = 0
        self.game_over = False
        self.spawn_interval = 60

    def draw_dino(self):
        pygame.draw.rect(self.screen, BLACK, (50, int(self.dino_y)+20, 40, 40))

    def draw_obstacles(self):
        for obs in self.obstacles:
            x, y, obs_type = obs
            if obs_type == "cactus":
                pygame.draw.rect(self.screen, GREEN, (int(x), SCREEN_HEIGHT - 40, 20, 40))
            elif obs_type == "pterodactyl":
                pygame.draw.rect(self.screen, RED, (int(x), y, 40, 20)) 

    def move_dino(self):
        if not self.game_over:
            self.dino_velocity += self.gravity
            self.dino_y += self.dino_velocity
            if self.dino_y > SCREEN_HEIGHT - 60:
                self.dino_y = SCREEN_HEIGHT - 60
                self.dino_velocity = 0

    def spawn_obstacle(self):
        if not self.game_over:
            self.obstacle_timer += 1
            if self.obstacle_timer >= self.spawn_interval:
                if random.random() < 0.5:
                    self.obstacles.append([SCREEN_WIDTH, SCREEN_HEIGHT - 40, "cactus"])
                else:
                    pterodactyl_y = random.choice([SCREEN_HEIGHT - 150, SCREEN_HEIGHT-100, SCREEN_HEIGHT])
                    self.obstacles.append([SCREEN_WIDTH, pterodactyl_y, "pterodactyl"])
                self.obstacle_timer = 0
                self.spawn_interval = random.randint(40, 60)

    def move_obstacles(self):
        if not self.game_over:
            for obs in self.obstacles[:]:
                obs[0] -= 5
                if obs[0] < -40: 
                    self.obstacles.remove(obs)

    def check_collision(self):
        dino_rect = pygame.Rect(50, int(self.dino_y)+20, 40, 40)
        for obs in self.obstacles:
            x, y, obs_type = obs
            if obs_type == "cactus":
                obs_rect = pygame.Rect(int(x), SCREEN_HEIGHT - 40, 20, 40)
            elif obs_type == "pterodactyl":
                obs_rect = pygame.Rect(int(x), y, 40, 20)
            if dino_rect.colliderect(obs_rect):
                self.game_over = True
                return True
        return False

    def render(self):
        self.screen.fill(WHITE)
        self.draw_dino()
        self.draw_obstacles()

        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {int(self.score)}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        if self.game_over:
            game_over_text = font.render("Game Over", True, BLACK)
            self.screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
        pygame.display.flip()

    def step(self, action):
        if not self.game_over:
            if action == "jump" and self.dino_y == SCREEN_HEIGHT - 60:
                self.dino_velocity = self.jump_power
            self.move_dino()
            self.spawn_obstacle()
            self.move_obstacles()
            self.score += 0.1
            self.check_collision()
        self.render()