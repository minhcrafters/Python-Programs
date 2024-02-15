import sys
import pygame
import random
import pygame_gui
import math

from itertools import product
from pygame import Vector2
from pg_utils import draw_text, scale_image

score = 0

SCALE_FACTOR = 0.75
WIDTH, HEIGHT = 800, 600

pygame.init()
pygame.font.init()
pygame.joystick.init()

controllers: dict[int, pygame.joystick.Joystick] = {}
controller_deadzone: float = 0.075

font = pygame.font.Font("./font/MinecraftBold.otf", 40)
smaller_font = pygame.font.Font("./font/MinecraftRegular.otf", 20)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Coin Collector")

manager = pygame_gui.UIManager((WIDTH, HEIGHT))

background = pygame.image.load("./img/background.png").convert()

player_image = pygame.image.load("./img/fox.png").convert_alpha()
coin_image = pygame.image.load("./img/coin.png").convert_alpha()
ball_image = pygame.image.load("./img/ball.png").convert_alpha()

background = scale_image(background, SCALE_FACTOR)
player_image = scale_image(player_image, SCALE_FACTOR)
coin_image = scale_image(coin_image, SCALE_FACTOR)
ball_image = scale_image(ball_image, 0.1 * SCALE_FACTOR)

clock = pygame.time.Clock()

coin_x = random.uniform(0, WIDTH / 2)
coin_y = random.uniform(0, HEIGHT / 2)

sprites = pygame.sprite.Group()

accel_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(0, 1),
    start_value=0.25,
    manager=manager,
)

speed_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=screen.get_rect(),
    value_range=(1, 100),
    start_value=20,
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
        self.vel = Vector2()
        self.accel = accel
        self.bouncy = 0.05
        self.has_switched_side = False
        self.rect = img.get_bounding_rect()

    @property
    def pos(self) -> Vector2:
        return Vector2(self.rect.centerx, self.rect.centery)

    @pos.setter
    def pos(self, pos: tuple[float | int, float | int]):
        self.rect.centerx, self.rect.centery = pos

    @pos.setter
    def pos(self, pos: Vector2):
        self.rect.centerx, self.rect.centery = pos.x, pos.y

    def control(self, controller_mode: bool = False):
        keys = pygame.key.get_pressed()

        if controller_mode:
            horizontal_input = (
                controllers[0].get_axis(0)
                if abs(controllers[0].get_axis(0)) >= controller_deadzone
                else 0
            )
            vertical_input = (
                controllers[0].get_axis(1)
                if abs(controllers[0].get_axis(1)) >= controller_deadzone
                else 0
            )
        else:
            horizontal_input = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
            vertical_input = int(keys[pygame.K_DOWN]) - int(keys[pygame.K_UP])

        self.vel.x += horizontal_input * self.accel * self.steps
        self.vel.y += vertical_input * self.accel * self.steps

        # print(self.vec.x)

        if controller_mode:
            if abs(controllers[0].get_axis(0)) <= controller_deadzone:
                if abs(self.vel.x) > 0.99:
                    self.vel.x *= 0.9
                else:
                    self.vel.x = 0
            if abs(controllers[0].get_axis(1)) <= controller_deadzone:
                if abs(self.vel.y) > 0.99:
                    self.vel.y *= 0.9
                else:
                    self.vel.y = 0
        else:
            if not (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]):
                if abs(self.vel.x) > 0.99:
                    self.vel.x *= 0.9
                else:
                    self.vel.x = 0
            if not (keys[pygame.K_UP] or keys[pygame.K_DOWN]):
                if abs(self.vel.y) > 0.99:
                    self.vel.y *= 0.9
                else:
                    self.vel.y = 0

    def move_rel(self, pos_rel: Vector2):
        if pos_rel.magnitude() > 0:
            self.vel.x += self.steps * self.accel * (pos_rel.x / pos_rel.magnitude())
            self.vel.y += self.steps * self.accel * (pos_rel.y / pos_rel.magnitude())

    def update(self):
        if self.vel.magnitude() >= self.steps:
            self.vel = self.vel.normalize() * self.steps

        self.vel *= 0.99

        self.rect.x += self.vel.x * SCALE_FACTOR
        self.rect.y += self.vel.y * SCALE_FACTOR

        if self.vel.x > self.steps:
            self.vel.x = -self.vel.x + self.bouncy * self.vel.x
            self.vel.x = self.steps

        if self.vel.y > self.steps:
            self.vel.y = -self.vel.y + self.bouncy * self.vel.y
            self.vel.y = self.steps

        self.rect.clamp_ip(screen.get_rect())

        if (self.vel.x < 0 and not self.has_switched_side) or (
            self.vel.x > 0 and self.has_switched_side
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
    draw_text(screen, font, text, pos=(text_x, text_y), shadow=True, shadow_offset=3)


def draw_timer(offset: int = 0):
    text = f"Accel: {round(accel_slider.get_current_value(), 2)} | Speed: {speed_slider.get_current_value()}"
    text_x = (WIDTH - smaller_font.size(text)[0]) // 2
    text_y = (HEIGHT - smaller_font.get_height()) - 80 + offset
    draw_text(
        screen,
        smaller_font,
        text,
        pos=(text_x, text_y),
        shadow=True,
    )


def draw_debug_menu(
    player: Actor,
    objects: list[Actor],
    auto_mode_enabled: bool,
    coin_collisions: int,
    coin_sprite: Actor,
    pos_rel: Vector2,
):
    debug_texts = [
        f"fps: {round(clock.get_fps(), 2)}",
        f"auto_mode: {auto_mode_enabled}",
        f"pos_player:\n{round(player.pos, 2)}",
        f"pos_coin:\n{round(coin_sprite.pos, 2)}",
        f"pos_rel:\n{pos_rel}",
        f"vel:\n{round(player.vel, 2)}",
        f"coins/sec: {round(coin_collisions, 1)}",
        f"{round(pos_rel.magnitude(), 1)}",
        f"{abs(pos_rel.x)}",
        f"{abs(pos_rel.y)}",
    ]

    for object in objects:
        pygame.draw.rect(screen, "red", object.rect, 1)

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
    global score, coin_x, coin_y, player_image, coin_image, background

    screen_width, screen_height = screen.get_size()
    background_width, background_height = background.get_size()

    coin_collisions = 0

    player = Actor(
        player_image,
        accel=accel_slider.get_current_value(),
        steps=speed_slider.get_current_value(),
    )

    player.rect.centerx = WIDTH // 2
    player.rect.centery = HEIGHT // 2

    coin = Actor(coin_image)

    # ball_obj = Actor(ball_image, steps=speed_slider.get_current_value())

    # ball_obj.rect.centerx = random.randint(0, WIDTH)
    # ball_obj.rect.centery = random.randint(0, HEIGHT)

    sprites.add(player)
    sprites.add(coin)
    # sprites.add(ball_obj)

    start_time = pygame.time.get_ticks()
    passed_time = 1

    auto_mode = False
    done_reset = True
    debug = False
    controls_hint = True
    _counter = 1000
    coin_cps = 0
    sliders_enabled = False
    controller_mode = False

    # thread = threading.Thread(
    #     target=predict, args=[create_dataset, model, player, coin, pos_rel]
    # )
    # thread.start()

    running = True
    while running:
        t = clock.tick(fps)
        dt = t / 1000.0

        # screen.fill((59, 177, 227))

        tiles_x = math.ceil(screen_width / background_width / SCALE_FACTOR)
        tiles_y = math.ceil(screen_height / background_height / SCALE_FACTOR)

        for x, y in product(range(tiles_x), range(tiles_y)):
            screen.blit(background, (x * background_width, y * background_height))

        pos_rel = Vector2(coin.rect.center) - Vector2(player.rect.center)

        accel_slider.set_current_value(player.accel)
        speed_slider.set_current_value(player.steps)

        # print(player.rect.right)

        # if player.rect.right <= 4:
        #     player.rect.x = WIDTH
        # if player.rect.left >= WIDTH:
        #     player.rect.right = 0
        # if player.rect.bottom <= 1:
        #     player.rect.top = HEIGHT
        # if player.rect.top >= HEIGHT:
        #     player.rect.bottom = 0

        if player.rect.left <= 0 or player.rect.right >= WIDTH:
            player.vel.x = -player.vel.x
        if player.rect.top <= 0 or player.rect.bottom >= HEIGHT:
            player.vel.y = -player.vel.y

        # if ball_obj.rect.left <= 0 or ball_obj.rect.right >= WIDTH:
        #     ball_obj.vel.x = -ball_obj.vel.x
        # if ball_obj.rect.top <= 0 or ball_obj.rect.bottom >= HEIGHT:
        #     ball_obj.vel.y = -ball_obj.vel.y

        if auto_mode:
            # mouse = pygame.mouse.get_pos()
            # player.move_rel(mouse[0], mouse[1])
            player.move_rel(pos_rel)
            # ball_obj.move_rel(pos_rel_ball)
            done_reset = False
        else:
            if not done_reset:
                if abs(player.vel.x) > 0.99:
                    player.vel.x *= 0.9
                else:
                    player.vel.x = 0

                if abs(player.vel.y) > 0.99:
                    player.vel.y *= 0.9
                else:
                    player.vel.y = 0
                done_reset = True
            player.control(controller_mode=controller_mode)

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

        normalized = (
            player.vel.normalize()
            if (player.vel.x, player.vel.y) > (0, 0)
            else player.vel
        )

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

            if event.type == pygame.JOYDEVICEADDED:
                joy = pygame.joystick.Joystick(event.device_index)
                controllers[joy.get_instance_id()] = joy
                print(f"Joystick {joy.get_instance_id()} connected")

            if event.type == pygame.JOYDEVICEREMOVED:
                if event.instance_id:
                    controller_mode = False
                    del controllers[event.instance_id]
                    print(f"Joystick {event.instance_id} disconnected")

            if event.type in (
                pygame.JOYAXISMOTION,
                pygame.JOYBALLMOTION,
                pygame.JOYBUTTONDOWN,
                pygame.JOYBUTTONUP,
                pygame.JOYHATMOTION,
            ):
                if (
                    abs(controllers.get(event.instance_id).get_axis(0))
                    >= controller_deadzone
                    and abs(controllers.get(event.instance_id).get_axis(1))
                    >= controller_deadzone
                ):
                    if not controller_mode:
                        controller_mode = True

            if event.type == pygame.JOYBUTTONDOWN:
                if controller_mode:
                    if event.button == 2:
                        debug = not debug
                    if event.button == 3:
                        auto_mode = not auto_mode
                    if event.button == 6:
                        controls_hint = not controls_hint
                    if event.button == 7:
                        player.rect.centerx = WIDTH // 2
                        player.rect.centery = HEIGHT // 2
                    if event.button == 11:
                        player.steps += 1
                    if event.button == 12:
                        player.steps -= 1
                    if event.button == 13:
                        player.accel -= 0.05
                    if event.button == 14:
                        player.accel += 0.05

            if event.type == pygame.KEYDOWN:
                if controller_mode:
                    controller_mode = False
                if event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                if event.key == pygame.K_F3:
                    debug = not debug
                if event.key == pygame.K_F2:
                    sliders_enabled = not sliders_enabled
                if event.key == pygame.K_TAB:
                    controls_hint = not controls_hint

                if event.key == pygame.K_r:
                    player.rect.centerx = WIDTH // 2
                    player.rect.centery = HEIGHT // 2

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == accel_slider:
                    player.accel = event.value
                if event.ui_element == speed_slider:
                    player.steps = event.value

            manager.process_events(event)

        # ball_obj.vel.y += 1

        # if player.rect.colliderect(ball_obj.rect):
        #     print("hit")
        #     pos = Vector2(ball_obj.rect.center) - Vector2(player.rect.center)
        #     v1 = ball_obj.vel.reflect(pos) * 2
        #     # v2 = player.vel.reflect(-pos)
        #     print(v1)
        #     ball_obj.vel = v1
        #     # player.vel = v2

        coin.rect.center = (coin_x, coin_y)
        # coin.rect.center = pygame.mouse.get_pos()

        if coin.rect.colliderect(player.rect):
            score += 1
            coin_collisions += 1
            coin_x = random.uniform(
                60,
                WIDTH - player.rect.x + (-20 if coin.rect.x > WIDTH // 2 else 20),
            )
            coin_y = random.uniform(
                60,
                HEIGHT - player.rect.y + (-20 if coin.rect.y > HEIGHT // 2 else 20),
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
        coin.update()
        # ball_obj.update()
        sprites.draw(screen)
        # screen.blit(coin, coin_rect)

        controls_hint_texts = [
            f"{'TAB' if not controller_mode else 'Options'}: Toggle controls",
            f"{'Arrow Keys' if not controller_mode else 'Left Stick'}: Move",
            f"{'R' if not controller_mode else 'Left Stick In'}: Reset position",
            # f"{'F2: Toggle sliders' if not controller_mode else 'D-pad U/D: Change speed{}D-pad L/R: Change acceleration'.format('\n')}",
            f"{'F3' if not controller_mode else 'Square'}: Toggle debug menu",
            f"{'Space' if not controller_mode else 'Triangle'}: Toggle Auto Mode",
            "ESC: Quit",
        ]

        if debug:
            draw_debug_menu(player, [player, coin], auto_mode, coin_cps, coin, pos_rel)
        elif controls_hint:
            draw_text(
                screen,
                smaller_font,
                "\n".join(controls_hint_texts),
                pos=(10, 10),
                shadow=True,
            )

        manager.update(dt)

        draw_scores()

        draw_timer(70 if not sliders_enabled else 0)

        if auto_mode:
            opacity_right = (
                128
                if player.vel.x < player.steps * normalized.x - 1
                else 255
                if player.vel.x > 0
                else 128
            )
            opacity_left = (
                128
                if player.vel.x > -(player.steps * normalized.x - 1)
                else 255
                if player.vel.x < 0
                else 128
            )
            opacity_down = (
                128
                if player.vel.y < player.steps * normalized.y - 1
                else 255
                if player.vel.y > 0
                else 128
            )
            opacity_up = (
                128
                if player.vel.y > -(player.steps * normalized.y - 1)
                else 255
                if player.vel.y < 0
                else 128
            )

            # the fix for a strange bug
            if player.vel.x < 0 and player.vel.y > 0:
                opacity_down = 255
                opacity_left = 255
            if player.vel.x > 0 and player.vel.y < 0:
                opacity_up = 255
                opacity_right = 255
        else:
            keys = pygame.key.get_pressed()
            opacity_right = (
                255
                if (
                    keys[pygame.K_RIGHT]
                    if not controller_mode
                    else 255
                    if controllers[0].get_axis(0) >= controller_deadzone
                    else 128
                )
                else 128
            )
            opacity_left = (
                255
                if (
                    keys[pygame.K_LEFT]
                    if not controller_mode
                    else 255
                    if controllers[0].get_axis(0) <= -controller_deadzone
                    else 128
                )
                else 128
            )
            opacity_down = (
                255
                if (
                    keys[pygame.K_DOWN]
                    if not controller_mode
                    else 255
                    if controllers[0].get_axis(1) >= controller_deadzone
                    else 128
                )
                else 128
            )
            opacity_up = (
                255
                if (
                    keys[pygame.K_UP]
                    if not controller_mode
                    else 255
                    if controllers[0].get_axis(1) <= -controller_deadzone
                    else 128
                )
                else 128
            )

        # print(player.vec.xy, opacity_down, opacity_left)

        right = draw_text(
            screen,
            smaller_font,
            text="Right",
            opacity=opacity_right,
            anchor="bottomleft",
            pos=(10, HEIGHT - 10),
            shadow=True,
        )

        left = draw_text(
            screen,
            smaller_font,
            text="Left",
            opacity=opacity_left,
            anchor="bottomleft",
            pos=(10, HEIGHT - 10 - right.get_height()),
            shadow=True,
        )

        down = draw_text(
            screen,
            smaller_font,
            text="Down",
            opacity=opacity_down,
            anchor="bottomleft",
            pos=(10, HEIGHT - 10 - right.get_height() - left.get_height()),
            shadow=True,
        )

        draw_text(
            screen,
            smaller_font,
            text="Up",
            opacity=opacity_up,
            anchor="bottomleft",
            pos=(
                10,
                HEIGHT
                - 10
                - right.get_height()
                - left.get_height()
                - down.get_height(),
            ),
            shadow=True,
        )

        # print(
        #     player.vel.normalize()
        #     if (player.vel.x, player.vel.y) != (0, 0)
        #     else player.vel
        # )

        if sliders_enabled:
            manager.draw_ui(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main(60)
