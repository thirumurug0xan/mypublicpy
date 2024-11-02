import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Endless Space Shooter")

# Clock to control the frame rate
clock = pygame.time.Clock()

# Load assets
ship_image = pygame.image.load('ship.png')
ship_rect = ship_image.get_rect()
bullet_image = pygame.image.load('bullet.png')
asteroid_image = pygame.image.load('asteroid.png')

# Ship class
class Ship:
    def __init__(self):
        self.image = ship_image
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        self.speed = 5
        self.bullets = []

    def move(self, dx):
        self.rect.x += dx * self.speed
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

    def shoot(self):
        bullet = Bullet(self.rect.centerx, self.rect.top)
        self.bullets.append(bullet)

    def draw(self):
        screen.blit(self.image, self.rect)
        for bullet in self.bullets:
            bullet.draw()

# Bullet class
class Bullet:
    def __init__(self, x, y):
        self.image = bullet_image
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = 10

    def move(self):
        self.rect.y -= self.speed

    def draw(self):
        screen.blit(self.image, self.rect)

# Asteroid class
class Asteroid:
    def __init__(self):
        self.image = asteroid_image
        self.rect = self.image.get_rect(center=(random.randint(0, SCREEN_WIDTH), 0))
        self.speed = random.randint(3, 6)

    def move(self):
        self.rect.y += self.speed

    def draw(self):
        screen.blit(self.image, self.rect)

# Main game loop
def main():
    ship = Ship()
    asteroids = []
    asteroid_timer = 0
    score = 0
    running = True

    while running:
        screen.fill(BLACK)
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    ship.shoot()

        # Ship movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            ship.move(-1)
        if keys[pygame.K_RIGHT]:
            ship.move(1)

        # Asteroid spawning
        if asteroid_timer == 0:
            asteroids.append(Asteroid())
            asteroid_timer = random.randint(20, 50)
        else:
            asteroid_timer -= 1

        # Bullet movement and collision
        for bullet in ship.bullets[:]:
            bullet.move()
            if bullet.rect.bottom < 0:
                ship.bullets.remove(bullet)

        # Asteroid movement and collision
        for asteroid in asteroids[:]:
            asteroid.move()
            if asteroid.rect.top > SCREEN_HEIGHT:
                asteroids.remove(asteroid)
            # Collision with bullets
            for bullet in ship.bullets:
                if bullet.rect.colliderect(asteroid.rect):
                    ship.bullets.remove(bullet)
                    asteroids.remove(asteroid)
                    score += 10
                    break

        # Drawing
        ship.draw()
        for asteroid in asteroids:
            asteroid.draw()

        # Display the score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()

        # Frame rate control
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
()