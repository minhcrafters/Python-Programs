"""
Reinforcement Unsupervised Learning Neural Network Model v2
Author: minhcrafters
With some code from StackOverflow :)
"""

import pygame
import nn_helper
import math
import os

from gymnasium import Env, spaces
import numpy as np

from itertools import product
from pygame import Vector2
from pg_utils import draw_text, scale_image
from collections.abc import MutableMapping

SCALE_FACTOR = 0.75
WIDTH, HEIGHT = 800, 600

SPEED = 25
ACCELERATION = 0.25


# class Actor(pygame.sprite.Sprite):
#     def __init__(
#         self, img: pygame.surface.Surface = None, steps: int = 12, accel: float = 0.15
#     ):
#         pygame.sprite.Sprite.__init__(self)
#         self.image = img if img is not None else None
#         SPEED = steps
#         self.vel = Vector2()
#         ACCELERATION = accel
#         self.bouncy = 0.05
#         self.has_switched_side = False
#         self.rect = img.get_bounding_rect() if img is not None else None

#     def set_image(self, img: pygame.surface.Surface):
#         self.image = img
#         self.rect = img.get_bounding_rect()

#     @property
#     def pos(self) -> Vector2:
#         return Vector2(self.rect.centerx, self.rect.centery)

#     @pos.setter
#     def pos(self, pos: tuple[float | int, float | int]):
#         self.rect.centerx, self.rect.centery = pos

#     @pos.setter
#     def pos(self, pos: Vector2):
#         self.rect.centerx, self.rect.centery = pos.x, pos.y

#     def control(self, horizontal_input: int, vertical_input: int):
#         self.vel.x += horizontal_input * ACCELERATION * SPEED
#         self.vel.y += vertical_input * ACCELERATION * SPEED

#         # print(self.vec.x)

#         if horizontal_input:
#             if abs(horizontal_input) <= 0.05:
#                 if abs(self.vel.x) > 0.99:
#                     self.vel.x *= 0.9
#                 else:
#                     self.vel.x = 0
#         if vertical_input:
#             if abs(vertical_input) <= 0.05:
#                 if abs(self.vel.y) > 0.99:
#                     self.vel.y *= 0.9
#                 else:
#                     self.vel.y = 0

#     def move_rel(self, pos_rel: Vector2):
#         if pos_rel.magnitude() > 0:
#             self.vel.x += SPEED * ACCELERATION * (pos_rel.x / pos_rel.magnitude())
#             self.vel.y += SPEED * ACCELERATION * (pos_rel.y / pos_rel.magnitude())

#     def update(self, screen: pygame.surface.Surface):
#         if self.vel.magnitude() >= SPEED:
#             self.vel = self.vel.normalize() * SPEED

#         self.vel *= 0.99

#         self.rect.x += self.vel.x * SCALE_FACTOR
#         self.rect.y += self.vel.y * SCALE_FACTOR

#         if self.vel.x > SPEED:
#             self.vel.x = -self.vel.x + self.bouncy * self.vel.x
#             self.vel.x = SPEED

#         if self.vel.y > SPEED:
#             self.vel.y = -self.vel.y + self.bouncy * self.vel.y
#             self.vel.y = SPEED

#         if (self.vel.x < 0 and not self.has_switched_side) or (
#             self.vel.x > 0 and self.has_switched_side
#         ):
#             self.image = pygame.transform.flip(self.image, True, False)
#             self.has_switched_side = not self.has_switched_side

#         darkened_image = self.image.copy()
#         darkened_image.fill((0, 0, 0, 128), None, pygame.BLEND_RGBA_MULT)

#         dropshadow_offset = 2 + (
#             self.image.get_width() // (self.image.get_width() / 1.5)
#         )

#         screen.blit(
#             darkened_image,
#             (
#                 self.rect.x + dropshadow_offset,
#                 self.rect.y + dropshadow_offset,
#             ),
#         )


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


class CoinCollectorEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        global SPEED

        super(CoinCollectorEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # move right, left, down, up
        self.observation_space = spaces.Dict(
            spaces={
                "player_pos": spaces.Box(
                    low=0,
                    high=WIDTH - 1,
                    shape=(2,),
                    dtype=int,
                ),
                "player_vel": spaces.Box(
                    low=-SPEED, high=SPEED, shape=(2,), dtype=np.float32
                ),
                "coin": spaces.Box(
                    low=0,
                    high=WIDTH - 1,
                    shape=(2,),
                    dtype=int,
                ),
            }
        )

        self.observation_space = spaces.flatten_space(self.observation_space)
        # print(self.observation_space)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.player = None
        self.dt = 0

        self.screen = None
        self.clock = None
        self._counter = 1000

        self.state = None
        self.timer = 0

        self._action_to_direction = {
            0: [1, 0],
            1: [-1, 0],
            2: [0, 1],
            3: [0, -1],
        }

        self.vel = Vector2()

        # Define any other necessary variables for the environment
        self.has_switched_side = False
        self.pos_rel = 0

    def draw_scores(self):
        text = f"{self.score}"
        text_x = (WIDTH - self.font.size(text)[0]) // 2
        text_y = self.font.get_height() // 2
        draw_text(
            self.screen,
            self.font,
            text,
            pos=(text_x, text_y),
            shadow=True,
            shadow_offset=3,
        )

    def flatten(self, dictionary: dict, parent_key="", separator="_"):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(self.flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def draw_timer(self, player, offset: int = 0):
        text = f"Accel: {round(ACCELERATION, 2)} | Speed: {SPEED}"
        text_x = (WIDTH - self.smaller_font.size(text)[0]) // 2
        text_y = (HEIGHT - self.smaller_font.get_height()) - 80 + offset
        draw_text(
            self.screen,
            self.smaller_font,
            text,
            pos=(text_x, text_y),
            shadow=True,
        )

    def _get_obs(self):
        # return np.stack((self.player_loc, self.vel, self.coin_loc))
        player_loc_list = self.player_loc.tolist()
        coin_loc_list = self.coin_loc.tolist()
        array = [
            player_loc_list[0],
            player_loc_list[1],
            coin_loc_list[0],
            coin_loc_list[1],
            self.vel.x,
            self.vel.y,
        ]
        return np.array(array, dtype=np.float64)
        # return (
        #     player_loc_list[0],
        #     player_loc_list[1],
        #     coin_loc_list[0],
        #     coin_loc_list[1],
        #     self.vel.x,
        #     self.vel.y,
        # )

    def _get_info(self):
        return {
            "rel_dist": self.pos_rel,
        }

    def _normalize(self, x, min_x, max_x):
        return (x - min_x) / (max_x - min_x)

    def step(self, action, curr_gen: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        done = False

        if self.timer <= 0:
            done = True

        # screen.fill((59, 177, 227))

        self.pos_rel = np.linalg.norm(self.coin_loc - self.player_loc)

        # self.player.pos.x = self.player_loc[0]
        # self.player.pos.y = self.player_loc[1]

        # print(player.rect.right)

        # if player.rect.right <= 4:
        #     player.rect.x = WIDTH
        # if player.rect.left >= WIDTH:
        #     player.rect.right = 0
        # if player.rect.bottom <= 1:
        #     player.rect.top = HEIGHT
        # if player.rect.top >= HEIGHT:
        #     player.rect.bottom = 0

        # if self.player.rect.left <= 0 or self.player.rect.right >= WIDTH:
        #     self.vel.x = -self.vel.x
        # if self.player.rect.top <= 0 or self.player.rect.bottom >= HEIGHT:
        #     self.vel.y = -self.vel.y

        # if ball_obj.rect.left <= 0 or ball_obj.rect.right >= WIDTH:
        #     ball_obj.vel.x = -ball_obj.vel.x
        # if ball_obj.rect.top <= 0 or ball_obj.rect.bottom >= HEIGHT:
        #     ball_obj.vel.y = -ball_obj.vel.y

        direction = self._action_to_direction[action]

        # self.player.control(*direction)

        self.vel.x += direction[0] * ACCELERATION * SPEED
        self.vel.y += direction[1] * ACCELERATION * SPEED

        # print(self.vec.x)

        if direction[0]:
            if abs(direction[0]) <= 0.05:
                if abs(self.vel.x) > 0.99:
                    self.vel.x *= 0.9
                else:
                    self.vel.x = 0
        if direction[1]:
            if abs(direction[1]) <= 0.05:
                if abs(self.vel.y) > 0.99:
                    self.vel.y *= 0.9
                else:
                    self.vel.y = 0

        if self.vel.magnitude() >= SPEED:
            self.vel = self.vel.normalize() * SPEED

        self.vel *= 0.99

        self.player_loc[0] += self.vel.x * SCALE_FACTOR
        self.player_loc[1] += self.vel.y * SCALE_FACTOR

        if self.vel.x > SPEED:
            self.vel.x = -self.vel.x + 0.9 * self.vel.x
            self.vel.x = SPEED

        if self.vel.y > SPEED:
            self.vel.y = -self.vel.y + 0.9 * self.vel.y
            self.vel.y = SPEED

        # coin_sprite.control()

        # coin_loc = self.coin_loc.tolist()
        # self.coin.rect.center = (coin_loc[0], coin_loc[1])
        # coin.rect.center = pygame.mouse.get_pos()

        # print(self.pos_rel)

        self.reward = (
            -self._normalize(float(self.pos_rel), 0, WIDTH) + self.score * 10 + 1
        )

        if self.pos_rel <= 38.5:
            self.score += 1
            self.coin_collisions += 1
            self.coin_loc = self.np_random.integers(0, HEIGHT - 20, size=2, dtype=int)

        # ball_obj.vel.y += 1

        # if player.rect.colliderect(ball_obj.rect):
        #     print("hit")
        #     pos = Vector2(ball_obj.rect.center) - Vector2(player.rect.center)
        #     v1 = ball_obj.vel.reflect(pos) * 2
        #     # v2 = player.vel.reflect(-pos)
        #     print(v1)
        #     ball_obj.vel = v1
        #     # player.vel = v2

        # self.coin_x = random.uniform(
        #     60,
        #     WIDTH
        #     - self.player.rect.x
        #     + (-20 if self.coin.rect.x > WIDTH // 2 else 20),
        # )
        # self.coin_y = random.uniform(
        #     60,
        #     HEIGHT
        #     - self.player.rect.y
        #     + (-20 if self.coin.rect.y > HEIGHT // 2 else 20),
        # )

        # if (
        #     coin_sprite.rect.centerx <= 10 or coin_sprite.rect.centerx >= WIDTH - 20
        # ) or (
        #     coin_sprite.rect.centery <= 10
        #     or coin_sprite.rect.centery >= HEIGHT - 20
        # ):
        #     coin_x = random.uniform(0, WIDTH - player.rect.x - 50)
        #     coin_y = random.uniform(0, HEIGHT - player.rect.y - 50)

        # ball_obj.update()

        observation = self._get_obs()
        info = self._get_info()

        # self.reward = nn_helper.calculate_reward(
        #     info["rel_dist"],
        #     self.coin_collisions,
        # )

        self._counter -= self.dt if self.render_mode == "human" else 16.67
        if self._counter < 0:
            self.timer -= 1
            self._counter += 1000

        if self.render_mode == "human":
            self._render_frame(curr_gen=curr_gen)

        return observation, self.reward, done, {}, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.score = 0
        self.timer = 20

        self.player_loc = np.array([WIDTH // 2, HEIGHT // 2])

        self.coin_loc = self.np_random.integers(0, HEIGHT - 20, size=2, dtype=int)

        # while np.array_equal(self.coin_loc, self.player_loc):
        #     self.coin_loc = self.np_random.integers(0, HEIGHT - 20, size=2, dtype=int)

        # self.current_state[2] = self.coin_loc
        # self.current_state[0] = self.player_loc

        # self.initial_vel = self.vel

        self.coin_collisions = 0

        # ball_obj = Actor(ball_image, steps=speed_slider.get_current_value())

        # ball_obj.rect.centerx = random.randint(0, WIDTH)
        # ball_obj.rect.centery = random.randint(0, HEIGHT)

        # sprites.add(ball_obj)

        self.auto_mode = False
        self.done_reset = True
        self.debug = False
        self._counter = 1000
        self.coin_cps = 0

        self.reward = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def draw_drop_shadow(
        self,
        screen: pygame.Surface,
        player_rect: pygame.Rect,
        player_image: pygame.Surface,
    ):
        darkened_image = player_image.copy()
        darkened_image.fill((0, 0, 0, 128), None, pygame.BLEND_RGBA_MULT)

        dropshadow_offset = 2 + (
            player_image.get_width() // (player_image.get_width() / 1.5)
        )

        screen.blit(
            darkened_image,
            (
                player_rect.x + dropshadow_offset,
                player_rect.y + dropshadow_offset,
            ),
        )

    def get_pos(self, pos: tuple[float | int, float | int]) -> Vector2:
        return Vector2(pos[0], pos[1])

    def render(self, curr_gen: int):
        return self._render_frame(curr_gen)

    def _render_frame(self, curr_gen: int):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()

            self.font = pygame.font.Font("./font/MinecraftBold.otf", 40)
            self.smaller_font = pygame.font.Font("./font/MinecraftRegular.otf", 20)
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

            pygame.display.set_caption("Coin Collector (Neural Network Version)")

            pygame.time.set_timer(pygame.USEREVENT, 1000)

            self.background = pygame.image.load("./img/background.png").convert()

            self.player_image = pygame.image.load("./img/fox.png").convert_alpha()
            self.coin_image = pygame.image.load("./img/coin.png").convert_alpha()

            self.background = scale_image(self.background, SCALE_FACTOR)
            self.player_image = scale_image(self.player_image, SCALE_FACTOR)
            self.coin_image = scale_image(self.coin_image, SCALE_FACTOR)

            self.screen_width, self.screen_height = self.screen.get_size()
            self.background_width, self.background_height = self.background.get_size()

            self.start_time = pygame.time.get_ticks()
            self.passed_time = 1
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            tiles_x = math.ceil(
                self.screen_width / self.background_width / SCALE_FACTOR
            )
            tiles_y = math.ceil(
                self.screen_height / self.background_height / SCALE_FACTOR
            )

            for x, y in product(range(tiles_x), range(tiles_y)):
                self.screen.blit(
                    self.background,
                    (x * self.background_width, y * self.background_height),
                )

            # self.player.rect.center = (self.player_loc[0], self.player_loc[1])
            # self.coin.rect.center = (self.coin_loc[0], self.coin_loc[1])
            self.player_rect = self.player_image.get_bounding_rect()
            self.coin_rect = self.coin_image.get_bounding_rect()

            self.player_rect.center = (self.player_loc[0], self.player_loc[1])
            self.coin_rect.center = (self.coin_loc[0], self.coin_loc[1])

            if (self.vel.x < 0 and not self.has_switched_side) or (
                self.vel.x > 0 and self.has_switched_side
            ):
                self.player_image = pygame.transform.flip(
                    self.player_image, True, False
                )
                self.has_switched_side = not self.has_switched_side

            self.player_rect.clamp_ip(self.screen.get_rect())

            self.player_loc = np.array(
                [self.player_rect.centerx, self.player_rect.centery]
            )

            self.draw_drop_shadow(self.screen, self.player_rect, self.player_image)
            self.draw_drop_shadow(self.screen, self.coin_rect, self.coin_image)

            self.screen.blit(self.player_image, self.player_rect)
            self.screen.blit(self.coin_image, self.coin_rect)

            # screen.blit(coin, coin_rect)

            debug_texts = [
                "pos_player:",
                f"{round(self.get_pos(self.player_rect.center), 2)}",
                "pos_coin:",
                f"{round(self.get_pos(self.coin_rect.center), 2)}",
                "vel:",
                f"{round(self.vel, 2)}",
                "pos_rel:",
                f"{round(self.pos_rel, 2)}",
                f"reward: {round(self.reward, 2)}",
                # f"{round(self.pos_rel, 1)}",
            ]

            # positions = [
            #     (WIDTH - 10, 10),
            #     (WIDTH - 10, 40 + self.smaller_font.get_height()),
            #     (WIDTH - 10, 50 + self.smaller_font.get_height()),
            #     (WIDTH - 10, 115 + self.smaller_font.get_height()),
            #     (WIDTH - 10, 180 + self.smaller_font.get_height()),
            #     # (WIDTH - 10, 245 + self.smaller_font.get_height()),
            # ]

            for i in range(len(debug_texts)):
                draw_text(
                    self.screen,
                    text=debug_texts[i],
                    font=self.smaller_font,
                    anchor="topright",
                    pos=(WIDTH - 10, 10 + self.smaller_font.get_height() * i * 1.235),
                    shadow=True,
                )

            self.draw_scores()

            self.draw_timer(self.player, 70)

            draw_text(
                self.screen,
                self.smaller_font,
                f"{'Gen #{}'.format(curr_gen)}\nTime left: {self.timer}",
                pos=(10, 10),
                shadow=True,
            )

            self.normalized = (
                self.vel.normalize() if (self.vel.x, self.vel.y) != (0, 0) else self.vel
            )

            opacity_right = (
                128
                if self.vel.x < SPEED * self.normalized.x - 1
                else 255
                if self.vel.x > 0
                else 128
            )
            opacity_left = (
                128
                if self.vel.x > -(SPEED * self.normalized.x - 1)
                else 255
                if self.vel.x < 0
                else 128
            )
            opacity_down = (
                128
                if self.vel.y < SPEED * self.normalized.y - 1
                else 255
                if self.vel.y > 0
                else 128
            )
            opacity_up = (
                128
                if self.vel.y > -(SPEED * self.normalized.y - 1)
                else 255
                if self.vel.y < 0
                else 128
            )

            # the fix for a strange bug
            if self.vel.x < 0 and self.vel.y > 0:
                opacity_down = 255
                opacity_left = 255
            if self.vel.x > 0 and self.vel.y < 0:
                opacity_up = 255
                opacity_right = 255

            # print(player.vec.xy, opacity_down, opacity_left)

            right = draw_text(
                self.screen,
                self.smaller_font,
                text="Right",
                opacity=opacity_right,
                anchor="bottomleft",
                pos=(10, HEIGHT - 10),
                shadow=True,
            )

            left = draw_text(
                self.screen,
                self.smaller_font,
                text="Left",
                opacity=opacity_left,
                anchor="bottomleft",
                pos=(10, HEIGHT - 10 - right.get_height()),
                shadow=True,
            )

            down = draw_text(
                self.screen,
                self.smaller_font,
                text="Down",
                opacity=opacity_down,
                anchor="bottomleft",
                pos=(10, HEIGHT - 10 - right.get_height() - left.get_height()),
                shadow=True,
            )

            draw_text(
                self.screen,
                self.smaller_font,
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
            pygame.event.pump()
            pygame.display.flip()
            self.dt = self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = CoinCollectorEnv(render_mode="human")
    # from gymnasium.utils.env_checker import check_env

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    n_episodes = 1000
    batch_size = 32

    agent = nn_helper.DQNAgent(n_episodes, state_size, action_size)

    # check_env(env)
    for e in range(n_episodes + 1):
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        time = 0
        while not done:
            env.render(e)
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action, e)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(
                    "Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(
                        e, n_episodes - 1, time, agent.epsilon
                    )
                )
            time += 1
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
        if e % 50 == 0:
            agent.save("./model/weights_" + "{:04d}".format(e) + ".hdf5")
    # model = nn_helper.create_model(env)
    # nn_helper.train_model(model, env)
    # nn_helper.evaluate_model(model, env)

    # model.save_weights(
    #     "./model/dqn_{}_weights.h5f".format("coin_collector"), overwrite=True
    # )
