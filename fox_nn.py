"""
Reinforcement Supervised Learning neural network model v1
Author: minhcrafters
With some code from StackOverflow :)
"""

import pygame
import random
import nn_helper
import math
import os

import pandas as pd

from datetime import datetime
from itertools import product
from pygame import Vector2
from pg_utils import draw_text, scale_image
from nn_helper import make_prediction

SCALE_FACTOR = 0.75
WIDTH, HEIGHT = 800, 600

SPEED = 25
ACCELERATION = 0.25


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

    def control(self, right=None, left=None, down=None, up=None):
        horizontal_input = right - left
        vertical_input = down - up

        self.vel.x += horizontal_input * self.accel * self.steps
        self.vel.y += vertical_input * self.accel * self.steps

        # print(self.vec.x)

        if right and left and down and up:
            if abs(horizontal_input) <= 0.05:
                if abs(self.vel.x) > 0.99:
                    self.vel.x *= 0.9
                else:
                    self.vel.x = 0
        if abs(vertical_input) <= 0.05:
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


def draw_timer(player: Actor, offset: int = 0):
    text = f"Accel: {round(player.accel, 2)} | Speed: {player.steps}"
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


def make_gif(frames_dir, delete_frames=True):
    from moviepy.editor import ImageSequenceClip
    from natsort import natsorted
    import glob

    frame_files = natsorted(glob.glob(os.path.join(frames_dir, "*.png")))

    clip = ImageSequenceClip(frame_files, fps=60)
    pics_dir = "./frames"
    os.makedirs(pics_dir, exist_ok=True)
    clip.write_videofile(os.path.join(pics_dir, "../test_nn.mp4"))
    if delete_frames:
        import shutil

        shutil.rmtree(frames_dir)


def predict(
    model, player: Actor, accel, coin: Actor, coin_cps: float, pos_rel: Vector2
):
    d = {
        "player_pos_x": player.rect.x,
        "player_pos_y": player.rect.y,
        "player_vel_x": player.vel.x,
        "player_vel_y": player.vel.y,
        "player_accel": accel,
        "coins_collected": score,
        "coins_per_sec": coin_cps,
        "coin_pos_x": coin.rect.x,
        "coin_pos_y": coin.rect.y,
        "rel_dist_x": pos_rel.x,
        "rel_dist_y": pos_rel.y,
    }
    prediction = make_prediction(model, pd.DataFrame({k: [v] for k, v in d.items()}))
    print(prediction)
    player.control(
        right=prediction[0],
        left=prediction[1],
        down=prediction[2],
        up=prediction[3],
    )


def main(
    seconds_to_eval: int,
    fps: int = 60,
    curr_gen: int = 0,
    model=None,
    record_frames: bool = False,
):
    pygame.init()
    pygame.font.init()
    pygame.joystick.init()

    global score, font, smaller_font, screen

    score = 0

    font = pygame.font.Font("./font/MinecraftBold.otf", 40)
    smaller_font = pygame.font.Font("./font/MinecraftRegular.otf", 20)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    pygame.display.set_caption("Coin Collector (Neural Network Version)")

    pygame.time.set_timer(pygame.USEREVENT, 1000)

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

    timer = seconds_to_eval

    screen_width, screen_height = screen.get_size()
    background_width, background_height = background.get_size()

    coin_collisions = 0

    player = Actor(
        player_image,
        accel=ACCELERATION,
        steps=SPEED,
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

    initial_vel = player.vel
    accel = 0

    frame_dir = "frames"
    os.makedirs(frame_dir, exist_ok=True)
    frame_num = 0

    auto_mode = False
    done_reset = True
    debug = False
    _counter = 1000
    coin_cps = 0
    sliders_enabled = False

    game_dataset = []

    pos_rel = Vector2(0, 0)
    # thread = threading.Thread(
    #     target=predict, args=[create_dataset, model, player, coin, pos_rel]
    # )
    # thread.start()
    running = True
    while running:
        t = clock.tick(fps)

        if timer <= 0:
            running = False

        # screen.fill((59, 177, 227))

        tiles_x = math.ceil(screen_width / background_width / SCALE_FACTOR)
        tiles_y = math.ceil(screen_height / background_height / SCALE_FACTOR)

        for x, y in product(range(tiles_x), range(tiles_y)):
            screen.blit(background, (x * background_width, y * background_height))

        pos_rel = Vector2(coin.rect.center) - Vector2(player.rect.center)

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
            if model:
                predict(model, player, accel, coin, coin_cps, pos_rel)
            else:
                auto_mode = True

        # coin_sprite.control()

        _counter -= t
        if _counter < 0:
            passed_time = pygame.math.clamp(
                passed_time, 1, (pygame.time.get_ticks() - start_time) / 1000
            )
            final_vel = player.vel
            coin_cps = coin_collisions / passed_time
            accel = (final_vel - initial_vel) / passed_time
            accel = math.sqrt(accel.x**2 + accel.y**2)
            initial_vel = final_vel
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

            if event.type == pygame.USEREVENT:
                timer -= 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F3:
                    debug = not debug

                if event.key == pygame.K_r:
                    player.rect.centerx = WIDTH // 2
                    player.rect.centery = HEIGHT // 2

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

        if debug:
            draw_debug_menu(player, [player, coin], auto_mode, coin_cps, coin, pos_rel)
        else:
            draw_text(
                screen,
                smaller_font,
                f"{'Gen #{}'.format(curr_gen) if curr_gen else 'Running traditional bot'}\nTime left: {timer}",
                pos=(10, 10),
                shadow=True,
            )

        draw_scores()

        draw_timer(player, 70 if not sliders_enabled else 0)

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

        # game_state = [fox_x, fox_y, coin_x, coin_y, relative_dist_x, relative_dist_y]
        # 0 - left, 1 - right
        # 2 - up, 3 - down
        # TODO: implement the data
        game_state = [
            player.rect.x,
            player.rect.y,
            player.vel.x,
            player.vel.y,
            accel,
            score,
            coin_cps,
            coin.rect.x,
            coin.rect.y,
            pos_rel.x,
            pos_rel.y,
        ]
        if player.vel.x != 0 or player.vel.y != 0:
            norm_x = abs(player.vel.normalize().x)
            norm_y = abs(player.vel.normalize().y)
            output_action = [
                norm_x if pos_rel.x > 0 else 0,
                norm_x if pos_rel.x < 0 else 0,
                norm_y if pos_rel.y > 0 else 0,
                norm_y if pos_rel.y < 0 else 0,
            ]
        else:
            output_action = [0, 0, 0, 0]

        game_dataset.append([game_state, output_action])

        if record_frames:
            pygame.image.save(screen, os.path.join(frame_dir, f"frame_{frame_num}.png"))
            frame_num += 1

        pygame.display.flip()

    pygame.quit()

    name = f"results_{player.steps}_{str(float(player.accel)).replace('.', '')}_{datetime.today().strftime('%d%m%Y')}_{datetime.today().strftime('%H%M%S')}.csv"

    with open(
        f"./dataset/{name}",
        "w",
    ) as f:
        f.write(
            "player_pos_x,player_pos_y,player_vel_x,player_vel_y,player_accel,coins_collected,coins_per_sec,coin_pos_x,coin_pos_y,rel_dist_x,rel_dist_y,move_right,move_left,move_down,move_up\n"
        )
        for data in game_dataset:
            f.write(",".join(map(str, data[0] + data[1])) + "\n")

    model = nn_helper.run(name, model if model else None)

    if record_frames:
        make_gif(frame_dir)

    return model, name


if __name__ == "__main__":
    gens = 10
    seconds_to_eval = 20

    result, name = main(seconds_to_eval, 60)
    for i in range(gens + 1):
        print(result)
        print(name)
        result, name = main(seconds_to_eval, 60, i, result)
    result.save("./model/model_{}.keras".format(name[8:-4]))