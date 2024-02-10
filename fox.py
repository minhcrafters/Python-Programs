import sys
import pygame
import random
import pygame_gui

score = 0

WIDTH, HEIGHT = 800, 600

pygame.init()
pygame.font.init()

font = pygame.font.SysFont("Consolas", 36)
smaller_font = pygame.font.SysFont("Consolas", 24)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Coin Collector")

manager = pygame_gui.UIManager((WIDTH, HEIGHT))

fox = pygame.image.load("./img/fox.png").convert_alpha()
coin = pygame.image.load("./img/coin.png").convert_alpha()
coin = pygame.transform.scale2x(coin)

clock = pygame.time.Clock()

coin_x = random.randint(0, WIDTH // 2)
coin_y = random.randint(0, HEIGHT // 2)

sprites = pygame.sprite.Group()

accel_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(0, 1),
    start_value=0.2,
    manager=manager,
)

speed_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(0, 40),
    start_value=16,
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


class Actor(pygame.sprite.Sprite):
    def __init__(
        self, img: pygame.surface.Surface, steps: int = 12, accel: float = 0.15
    ):
        pygame.sprite.Sprite.__init__(self)
        self.image = img
        self.steps = steps
        self.vec = pygame.Vector2()
        self.accel = accel
        self.has_switched_side = False
        self.rect = img.get_bounding_rect()

    @property
    def pos(self) -> pygame.Vector2:
        return pygame.Vector2(self.rect.centerx, self.rect.centery)

    @pos.setter
    def pos(self, pos: tuple[float | int, float | int]):
        self.rect.centerx, self.rect.centery = pos

    @pos.setter
    def pos(self, pos: pygame.Vector2):
        self.rect.centerx, self.rect.centery = pos.x, pos.y

    def control(self):
        keys = pygame.key.get_pressed()

        horizontal_input = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        vertical_input = int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])

        self.vec.x += horizontal_input * self.accel * self.steps
        self.vec.y += vertical_input * self.accel * self.steps

        # print(self.vec.x)

        if not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
            if abs(self.vec.x) > 0.9:
                self.vec.x *= 0.9
            else:
                self.vec.x = 0
        if not (keys[pygame.K_UP] or keys[pygame.K_DOWN]):
            if abs(self.vec.y) > 0.9:
                self.vec.y *= 0.9
            else:
                self.vec.y = 0

    def move_rel(self, pos_rel: pygame.Vector2):
        if pos_rel.magnitude() > 0:
            self.vec.x += self.steps * self.accel * pos_rel.x / pos_rel.magnitude()
            self.vec.y += self.steps * self.accel * pos_rel.y / pos_rel.magnitude()

    def update(self):
        if self.vec.magnitude() >= self.steps:
            self.vec = self.vec.normalize() * self.steps

        self.rect.x += self.vec.x
        self.rect.y += self.vec.y

        self.vec *= 0.99

        if (self.vec.x < 0 and not self.has_switched_side) or (
            self.vec.x > 0 and self.has_switched_side
        ):
            self.image = pygame.transform.flip(self.image, True, False)
            self.has_switched_side = not self.has_switched_side


def draw_scores():
    score_text = font.render("Score: " + str(score), True, "white")
    text_x = (WIDTH - score_text.get_width()) // 2
    text_y = font.get_height() // 2
    screen.blit(score_text, (text_x, text_y))


def draw_timer():
    time_text = smaller_font.render(
        "Accel: "
        + str(round(accel_slider.get_current_value(), 2))
        + " | Speed: "
        + str(speed_slider.get_current_value()),
        True,
        "white",
    )
    text_x = (WIDTH - time_text.get_width()) // 2
    text_y = (HEIGHT - smaller_font.get_height()) - 80
    screen.blit(time_text, (text_x, text_y))


def draw_debug_menu(
    player: Actor, auto_mode_enabled: bool, coin_sprite: Actor, pos_rel: pygame.Vector2
):
    screen.blit(
        smaller_font.render(
            f"fps: {str(round(clock.get_fps(), 2))}", True, (255, 255, 255)
        ),
        (10, 10),
    )

    screen.blit(
        smaller_font.render(
            f"auto_mode: {str(auto_mode_enabled)}", True, (255, 255, 255)
        ),
        (10, 41),
    )

    screen.blit(
        smaller_font.render(
            f"pos_fox:\n{str(round(player.pos, 2))}", True, (255, 255, 255)
        ),
        (10, 50 + smaller_font.get_height()),
    )

    screen.blit(
        smaller_font.render(
            f"pos_coin:\n{str(round(coin_sprite.pos, 2))}",
            True,
            (255, 255, 255),
        ),
        (10, 115 + smaller_font.get_height()),
    )

    screen.blit(
        smaller_font.render(f"pos_rel:\n{str(pos_rel)}", True, (255, 255, 255)),
        (10, 180 + smaller_font.get_height()),
    )

    screen.blit(
        smaller_font.render(
            f"vel:\n{str(round(player.vec, 2))}", True, (255, 255, 255)
        ),
        (10, 245 + smaller_font.get_height()),
    )

    screen.blit(
        smaller_font.render(str(round(pos_rel.magnitude(), 1)), True, (255, 255, 255)),
        pygame.draw.line(
            screen, (255, 0, 0), player.rect.center, coin_sprite.rect.center
        ).center,
    )

    screen.blit(
        smaller_font.render(str(abs(round(pos_rel.x, 1))), True, (255, 255, 255)),
        pygame.draw.line(
            screen,
            (255, 0, 0),
            player.rect.center,
            (coin_sprite.rect.centerx, player.rect.centery),
        ).center,
    )

    screen.blit(
        smaller_font.render(str(abs(round(pos_rel.y, 1))), True, (255, 255, 255)),
        pygame.draw.line(
            screen,
            (255, 0, 0),
            coin_sprite.rect.center,
            (coin_sprite.rect.centerx, player.rect.centery),
        ).center,
    )


def main(fps: int = 60):
    global score, coin_x, coin_y, fox, coin

    player = Actor(fox)
    player.rect.centerx = WIDTH // 2
    player.rect.centery = HEIGHT // 2

    coin_sprite = Actor(coin)

    sprites.add(player)
    sprites.add(coin_sprite)

    running = True
    auto_mode = False
    done_reset = True
    debug = False
    while running:
        dt = clock.tick(fps) / 1000.0

        screen.fill((59, 177, 227))

        pos_rel = pygame.Vector2(coin_sprite.rect.center) - pygame.Vector2(
            player.rect.center
        )

        if auto_mode:
            # mouse = pygame.mouse.get_pos()
            # player.move_rel(mouse[0], mouse[1])
            player.move_rel(pos_rel)
            done_reset = False
        else:
            if not done_reset:
                if abs(player.vec.x) > 0.9:
                    player.vec.x *= 0.9
                else:
                    player.vec.x = 0

                if abs(player.vec.y) > 0.9:
                    player.vec.y *= 0.9
                else:
                    player.vec.y = 0
                done_reset = True
            player.control()

        # coin_sprite.control()

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                if event.key == pygame.K_F1:
                    debug = not debug
                if event.key == pygame.K_F2:
                    player.rect.centerx = WIDTH // 2
                    player.rect.centery = HEIGHT // 2

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == accel_slider:
                    player.accel = event.value
                if event.ui_element == speed_slider:
                    player.steps = event.value

            manager.process_events(event)

        coin_sprite.rect.topleft = (coin_x, coin_y)

        if coin_sprite.rect.colliderect(player.rect):
            score += 1
            coin_x = random.uniform(10, WIDTH - player.rect.x + 10)
            coin_y = random.uniform(10, HEIGHT - player.rect.y + 10)
            # if (
            #     coin_sprite.rect.centerx <= 10 or coin_sprite.rect.centerx >= WIDTH - 20
            # ) or (
            #     coin_sprite.rect.centery <= 10
            #     or coin_sprite.rect.centery >= HEIGHT - 20
            # ):
            #     coin_x = random.uniform(0, WIDTH - player.rect.x - 50)
            #     coin_y = random.uniform(0, HEIGHT - player.rect.y - 50)

        player.update()
        coin_sprite.update()
        sprites.draw(screen)
        # screen.blit(coin, coin_rect)

        if debug:
            draw_debug_menu(player, auto_mode, coin_sprite, pos_rel)

        manager.update(dt)

        draw_scores()
        draw_timer()

        manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main(60)
