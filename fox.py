import sys
import pygame
import random
import pygame_gui

score = 0

WIDTH, HEIGHT = 800, 600

pygame.init()
pygame.font.init()

font = pygame.font.SysFont("Consolas", 36)
screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=1)

pygame.display.set_caption("Coin Collector")

manager = pygame_gui.UIManager((WIDTH, HEIGHT))

fox = pygame.image.load("./img/fox.png").convert_alpha()
coin = pygame.image.load("./img/coin.png").convert_alpha()

clock = pygame.time.Clock()

coin_x = random.randint(0, WIDTH // 2)
coin_y = random.randint(0, HEIGHT // 2)

coin_rect = coin.get_rect()
player_list = pygame.sprite.Group()

accel_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(0, 2.5),
    start_value=0.15,
    manager=manager,
)

speed_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(1, 40),
    start_value=12,
    manager=manager,
)


accel_slider.set_dimensions((500, 30))
accel_slider.set_position(
    (
        (WIDTH - accel_slider.rect.width) / 2,
        (HEIGHT - accel_slider.rect.height) - 46,
    )
)

speed_slider.set_dimensions((500, 30))
speed_slider.set_position(
    (
        (WIDTH - accel_slider.rect.width) / 2,
        (HEIGHT - accel_slider.rect.height) - 12,
    )
)


class Player(pygame.sprite.Sprite):
    def __init__(
        self, img: pygame.surface.Surface, steps: int = 12, accel: float = 0.15
    ):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.steps = steps
        self.vec = pygame.Vector2()
        self.accel = accel
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
            dx *= 0.92

        if keys[pygame.K_UP]:
            dy += -self.steps
            self.facing = "up"
        elif keys[pygame.K_DOWN]:
            dy += self.steps
            self.facing = "down"
        else:
            dy *= 0.92

        clamped_pos_x = pygame.math.clamp(dx * self.accel, -self.steps, self.steps)
        clamped_pos_y = pygame.math.clamp(dy * self.accel, -self.steps, self.steps)
        self.vec.x += clamped_pos_x
        self.vec.y += clamped_pos_y

    def move_to(self, x: int, y: int):
        pos_rel = pygame.Vector2(x, y) - pygame.Vector2(self.rect.center)
        print(pos_rel)
        dx = pos_rel.x / self.steps
        dy = pos_rel.y / self.steps
        clamped_pos_rel_x = pygame.math.clamp(dx * self.accel, -self.steps, self.steps)
        clamped_pos_rel_y = pygame.math.clamp(dy * self.accel, -self.steps, self.steps)
        self.vec.x += clamped_pos_rel_x
        self.vec.y += clamped_pos_rel_y

    def update(self):
        if self.vec.magnitude() >= self.steps:
            self.vec = self.vec.normalize() * self.steps

        self.rect.x += self.vec.x
        self.rect.y += self.vec.y

        # self.rect.clamp_ip(screen.get_rect())


def draw_scores():
    score_text = font.render("Score: " + str(score), True, "white")
    text_x = (WIDTH - score_text.get_width()) // 2
    text_y = font.get_height() // 2 + 10
    screen.blit(score_text, (text_x, text_y))


def draw_timer():
    time_text = font.render(
        "Accel: "
        + str(round(accel_slider.get_current_value(), 2))
        + " | Speed: "
        + str(speed_slider.get_current_value()),
        True,
        "white",
    )
    text_x = (WIDTH - time_text.get_width()) // 2
    text_y = (HEIGHT - font.get_height()) - 100
    screen.blit(time_text, (text_x, text_y))


def main():
    global score, coin_x, coin_y, coin_rect, fox, coin

    time_delta = clock.tick(60) / 1000.0

    player = Player(fox, steps=12)
    player.rect.centerx = WIDTH // 2
    player.rect.centery = HEIGHT // 2

    player_list.add(player)

    running = True
    auto_move = False
    done_reset = True
    debug = False
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

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_move = not auto_move
                if event.key == pygame.K_F1:
                    debug = not debug

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == accel_slider:
                    player.accel = event.value
                if event.ui_element == speed_slider:
                    player.steps = event.value

            manager.process_events(event)

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

        if debug:
            pygame.draw.line(screen, (255, 0, 0), player.rect.center, coin_rect.center)

        manager.update(time_delta)

        draw_scores()
        draw_timer()

        manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
