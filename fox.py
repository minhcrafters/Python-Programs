import sys
import pygame
import random
import pygame_gui
import math

from itertools import product
from pg_utils import draw_text, scale_image

score = 0

SCALE_FACTOR = 0.75
WIDTH, HEIGHT = 800, 600

pygame.init()
pygame.font.init()

font = pygame.font.Font("./font/MinecraftBold.otf", 36)
smaller_font = pygame.font.Font("./font/MinecraftRegular.otf", 20)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Coin Collector")

manager = pygame_gui.UIManager((WIDTH, HEIGHT))

background = pygame.image.load("./img/background.png").convert()

fox = pygame.image.load("./img/fox.png").convert_alpha()
coin = pygame.image.load("./img/coin.png").convert_alpha()

background = scale_image(background, SCALE_FACTOR)
fox = scale_image(fox, SCALE_FACTOR)
coin = scale_image(coin, SCALE_FACTOR)

clock = pygame.time.Clock()

coin_x = random.randint(0, WIDTH // 2)
coin_y = random.randint(0, HEIGHT // 2)

pygame.time.set_timer(pygame.USEREVENT, 1000)

sprites = pygame.sprite.Group()

accel_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(0, 1),
    start_value=0.2,
    manager=manager,
)

speed_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(0, 100),
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

        self.rect.x += self.vec.x * SCALE_FACTOR
        self.rect.y += self.vec.y * SCALE_FACTOR

        self.vec *= 0.99

        self.rect.clamp_ip(screen.get_rect())

        if (self.vec.x < 0 and not self.has_switched_side) or (
            self.vec.x > 0 and self.has_switched_side
        ):
            self.image = pygame.transform.flip(self.image, True, False)
            self.has_switched_side = not self.has_switched_side

        darkened_image = self.image.copy()
        darkened_image.fill((0, 0, 0, 128), None, pygame.BLEND_RGBA_MULT)

        dropshadow_offset = 2 + (
            self.image.get_width() // (self.image.get_width() / 1.5)
        )

        screen.blit(
            darkened_image,
            (self.rect.x + dropshadow_offset, self.rect.y + dropshadow_offset),
        )


def draw_scores():
    text = f"{score}"
    text_x = (WIDTH - font.size(text)[0]) // 2
    text_y = font.get_height() // 2
    draw_text(screen, font, text, pos=(text_x, text_y), shadow=True, shadow_offset=2)


def draw_timer(offset: int = 0):
    text = f"Accel: {round(accel_slider.get_current_value(), 2)} | Speed: {speed_slider.get_current_value()}"
    text_x = (WIDTH - smaller_font.size(text)[0]) // 2
    text_y = (HEIGHT - smaller_font.get_height()) - 80 + offset
    draw_text(
        screen,
        smaller_font,
        f"Accel: {round(accel_slider.get_current_value(), 2)} | Speed: {speed_slider.get_current_value()}",
        pos=(text_x, text_y),
        shadow=True,
    )


def draw_debug_menu(
    player: Actor,
    auto_mode_enabled: bool,
    coin_collisions: int,
    coin_sprite: Actor,
    pos_rel: pygame.Vector2,
):
    debug_texts = [
        f"fps: {round(clock.get_fps(), 2)}",
        f"auto_mode: {auto_mode_enabled}",
        f"pos_fox:\n{round(player.pos, 2)}",
        f"pos_coin:\n{round(coin_sprite.pos, 2)}",
        f"pos_rel:\n{pos_rel}",
        f"vel:\n{round(player.vec, 2)}",
        f"coins/sec: {round(coin_collisions, 1)}",
        f"{round(pos_rel.magnitude(), 1)}",
        f"{abs(pos_rel.x)}",
        f"{abs(pos_rel.y)}",
    ]

    x_line = pygame.draw.line(
        screen,
        (255, 0, 0),
        player.rect.center,
        (coin_sprite.rect.centerx, player.rect.centery),
    )

    y_line = pygame.draw.line(
        screen,
        (255, 0, 0),
        coin_sprite.rect.center,
        (coin_sprite.rect.centerx, player.rect.centery),
    )

    positions = [
        (10, 10),
        (10, 40),
        (10, 50 + smaller_font.get_height()),
        (10, 115 + smaller_font.get_height()),
        (10, 180 + smaller_font.get_height()),
        (10, 245 + smaller_font.get_height()),
        (10, 310 + smaller_font.get_height()),
        pygame.draw.line(
            screen, (255, 0, 0), player.rect.center, coin_sprite.rect.center
        ).center,
        (
            x_line.centerx - smaller_font.size(f"{abs(pos_rel.x)}")[0] / 2,
            x_line.centery,
        ),
        (
            y_line.centerx - smaller_font.size(f"{abs(pos_rel.y)}")[0],
            y_line.centery - smaller_font.size(f"{abs(pos_rel.y)}")[1] / 2,
        ),
    ]

    for text, pos in zip(debug_texts, positions):
        draw_text(
            screen,
            text=text,
            font=smaller_font,
            pos=pos,
            shadow=True,
        )


def main(fps: int = 60):
    global score, coin_x, coin_y, fox, coin, background, SCALE_FACTOR

    coin_collisions = 0

    player = Actor(fox)
    player.rect.centerx = WIDTH // 2
    player.rect.centery = HEIGHT // 2

    coin_sprite = Actor(coin)

    sprites.add(player)
    sprites.add(coin_sprite)

    start_time = pygame.time.get_ticks()
    passed_time = 1

    running = True
    auto_mode = False
    done_reset = True
    debug = False
    _counter = 1000
    coin_cps = 0
    sliders_enabled = False

    while running:
        t = clock.tick(fps)
        dt = t / 1000.0

        # screen.fill((59, 177, 227))
        screen_width, screen_height = screen.get_size()
        background_width, background_height = background.get_size()

        tiles_x = math.ceil(screen_width / background_width / SCALE_FACTOR)
        tiles_y = math.ceil(screen_height / background_height / SCALE_FACTOR)

        for x, y in product(range(tiles_x), range(tiles_y)):
            screen.blit(background, (x * background_width, y * background_height))

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

        _counter -= t
        if _counter < 0:
            passed_time = pygame.math.clamp(
                passed_time, 1, (pygame.time.get_ticks() - start_time) / 1000
            )
            coin_cps = coin_collisions / passed_time
            coin_collisions = 0
            start_time = pygame.time.get_ticks()
            _counter += 1000

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                if event.key == pygame.K_F3:
                    debug = not debug
                if event.key == pygame.K_F2:
                    sliders_enabled = not sliders_enabled

                if event.key == pygame.K_r:
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
            coin_collisions += 1
            coin_x = random.uniform(
                60,
                WIDTH
                - player.rect.x
                + (-20 if coin_sprite.rect.x > WIDTH // 2 else 20),
            )
            coin_y = random.uniform(
                60,
                HEIGHT
                - player.rect.y
                + (-20 if coin_sprite.rect.y > HEIGHT // 2 else 20),
            )
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
            draw_debug_menu(player, auto_mode, coin_cps, coin_sprite, pos_rel)
        else:
            draw_text(
                screen,
                smaller_font,
                "Arrow Keys: Move\nF1: Toggle debug menu\nF2: Toggle sliders\nSpace: Toggle Auto Mode\nR: Reset position\nESC: Quit",
                pos=(10, 10),
                shadow=True,
            )

        manager.update(dt)

        draw_scores()

        draw_timer(70 if not sliders_enabled else 0)

        if sliders_enabled:
            manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main(60)
