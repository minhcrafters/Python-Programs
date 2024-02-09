import sys
import pygame
import random

score = 0
timer = 0

WIDTH = HEIGHT = 600

pygame.init()
pygame.font.init()

font = pygame.font.SysFont("Consolas", 36)
screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=1)

pygame.display.set_caption("Coin Collector")

fox = pygame.image.load("./img/fox.png").convert_alpha()
coin = pygame.image.load("./img/coin.png").convert_alpha()

clock = pygame.time.Clock()
pygame.time.set_timer(pygame.USEREVENT, 1000)

coin_x = random.randint(0, WIDTH // 2)
coin_y = random.randint(0, HEIGHT // 2)

coin_rect = coin.get_rect()
player_list = pygame.sprite.Group()


class Player(pygame.sprite.Sprite):
    def __init__(self, img: pygame.surface.Surface, steps: int = 12):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.steps = steps
        self.vec = pygame.Vector2()
        self.rect = img.get_bounding_rect()

    def control(self):
        keys = pygame.key.get_pressed()

        dx = 0
        dy = 0

        if keys[pygame.K_LEFT]:
            dx += -self.steps
            self.facing = "left"
        elif keys[pygame.K_RIGHT]:
            dx += self.steps
            self.facing = "right"
        else:
            dx = 0

        if keys[pygame.K_UP]:
            dy += -self.steps
            self.facing = "up"
        elif keys[pygame.K_DOWN]:
            dy += self.steps
            self.facing = "down"
        else:
            dy = 0

        self.vec.x = min(self.steps, max(-self.steps, dx))
        self.vec.y = min(self.steps, max(-self.steps, dy))

    def move_to(self, x: int, y: int):
        pos_rel = pygame.Vector2(x, y) - pygame.Vector2(self.rect.center)
        print(pos_rel)
        self.vec.x = min(self.steps, max(-self.steps, pos_rel.x))
        self.vec.y = min(self.steps, max(-self.steps, pos_rel.y))

    def update(self):
        if self.vec.magnitude() >= self.steps:
            self.vec = self.vec.normalize() * self.steps

        self.rect.x += self.vec.x
        self.rect.y += self.vec.y


def draw_scores():
    score_text = font.render("Score: " + str(score), True, "white")
    text_x = (WIDTH - score_text.get_width()) // 2
    text_y = (HEIGHT - font.get_height() // 2) - font.get_height()
    screen.blit(score_text, (text_x, text_y))


def draw_timer():
    time_text = font.render("Time: " + str(timer), True, "white")
    text_x = (WIDTH - time_text.get_width()) // 2
    text_y = font.get_height() // 2
    screen.blit(time_text, (text_x, text_y))


def main():
    global score, coin_x, coin_y, coin_rect, fox, coin, timer

    player = Player(fox, steps=12)
    player.rect.centerx = WIDTH // 2
    player.rect.centery = HEIGHT // 2

    player_list.add(player)

    running = True
    auto_move = False
    done_reset = True
    while running:
        screen.fill((59, 177, 227))

        if auto_move:
            # mouse = pygame.mouse.get_pos()
            # player.move_to(mouse[0], mouse[1])
            player.move_to(coin_rect.x, coin_rect.y)
            done_reset = False
        elif not auto_move:
            if not done_reset:
                player.vec.x = 0
                player.vec.y = 0
                done_reset = True
            player.control()

        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                timer += 1
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_move = not auto_move

        coin_rect.x = coin_x
        coin_rect.y = coin_y

        if coin_rect.colliderect(player.rect):
            score += 1
            coin_x = random.uniform(0, WIDTH - player.rect.x / 2)
            coin_y = random.uniform(0, HEIGHT - player.rect.y / 2)
            if (coin_rect.center[0] <= 10 or coin_rect.center[0] >= WIDTH - 20) or (
                coin_rect.center[1] <= 10 or coin_rect.center[1] >= HEIGHT - 20
            ):
                coin_x = random.uniform(0, WIDTH - coin_x - 50)
                coin_y = random.uniform(0, HEIGHT - coin_y - 50)

        player.update()
        player_list.draw(screen)
        screen.blit(coin, coin_rect)

        draw_scores()
        draw_timer()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
