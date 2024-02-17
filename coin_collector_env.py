"""
Reinforcement Unsupervised Learning Neural Network Model v2
Author: minhcrafters
With some code from StackOverflow :)
"""

import pygame
import math
import os

from gymnasium import Env, spaces
from nn_helper import calculate_reward
import numpy as np
from itertools import product
from pygame import Vector2
from pg_utils import draw_text, scale_image
from threading import Thread, Event


class Repeat(Thread):
    def __init__(self, delay, function, *args, **kwargs):
        """
        Initialize the thread with the given delay, function, args, and kwargs.
        """
        Thread.__init__(self)
        self.abort = Event()
        self.delay = delay
        self.args = args
        self.kwargs = kwargs
        self.function = function

    def stop(self):
        self.abort.set()

    def run(self):
        while not self.abort.is_set():
            self.function(*self.args, **self.kwargs)
            self.abort.wait(self.delay)


SCALE_FACTOR = 0.75
WIDTH = HEIGHT = 800

SPEED = 25
ACCELERATION = 0.25


def make_gif(frames_dir, delete_frames=True):
    """
    Create a GIF from a directory of image frames.

    Args:
        frames_dir (str): The directory containing the image frames.
        delete_frames (bool, optional): Whether to delete the image frames after creating the GIF. Defaults to True.

    Returns:
        None
    """
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
        """
        Constructor for CoinCollectorEnv.

        Args:
            render_mode (str, optional): The rendering mode. Defaults to None.

        Returns:
            None
        """
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

        self.current_step = 0

        self.repeat_every = None

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
        """
        This function draws the score on the screen using the provided font and position.
        """
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

    def draw_timer(self, player, offset: int = 0):
        """
        Draws a timer on the screen for the specified player.

        Args:
            player: The player for whom the timer is being drawn.
            offset (int): The vertical offset for the timer display.

        Returns:
            None
        """
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
        """
        Get observation data for the player including player location, velocity, and coin location.

        Returns:
            NDArray: A Numpy array containing player location, coin location, and velocity.
        """
        # return np.stack((self.player_loc, self.vel, self.coin_loc))
        player_loc_list = self.player_loc.tolist()
        coin_loc_list = self.coin_loc.tolist()

        # Scale player and coin location data between 0-1
        player_loc_scaled = [player_loc_list[0] / WIDTH, player_loc_list[1] / WIDTH]
        coin_loc_scaled = [coin_loc_list[0] / WIDTH, coin_loc_list[1] / WIDTH]

        # Scale velocity components between 0-1
        vel_scaled = [self.vel.x / SPEED, self.vel.y / SPEED]

        # Combine all scaled data into an array
        frame = np.array(
            [
                player_loc_scaled[0],
                player_loc_scaled[1],
                coin_loc_scaled[0],
                coin_loc_scaled[1],
                vel_scaled[0],
                vel_scaled[1],
            ]
        )

        return frame

    def _get_info(self):
        return {
            "rel_dist": self.pos_rel,
            "coins_collected": self.score,
        }

    def _normalize(self, x, min_x, max_x):
        return (x - min_x) / (max_x - min_x)

    def _decrease_timer(self):
        self.timer -= 1

    def step(self, action, curr_gen: int):
        """
        A function to perform a step in the game environment.

        Parameters:
            action: The action to be taken.
            curr_gen: The current generation.

        Returns:
            observation: The observation of the environment after the step.
            reward: The reward obtained from the step.
            done: Indicates if the episode is finished.
            info: Additional information about the step.
        """
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        done = False

        if self.timer <= 0:
            done = True

        self.current_step += 1

        direction = self._action_to_direction[action]

        self.vel.x += direction[0] * ACCELERATION * SPEED
        self.vel.y += direction[1] * ACCELERATION * SPEED

        self.pos_rel = np.linalg.norm(self.coin_loc - self.player_loc)

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

        prev_player_loc = self.player_loc

        self.player_loc[0] += self.vel.x * SCALE_FACTOR
        self.player_loc[1] += self.vel.y * SCALE_FACTOR

        self.player_loc[0] = sorted((0, self.player_loc[0], WIDTH - 1))[1]
        self.player_loc[1] = sorted((0, self.player_loc[1], HEIGHT - 1))[1]

        self.reward = calculate_reward(
            self.player_loc,
            self.coin_loc,
            self.score,
            curr_gen + 1,
            previous_agent_position=prev_player_loc,
        )

        # self.reward = 1.0 / self.pos_rel + self.coin_collisions * (curr_gen + 1)

        if self.pos_rel <= 38.5:
            self.score += 1
            self.coin_loc = self.np_random.integers(0, HEIGHT - 20, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" and self.screen:
            if pygame.event.get(pygame.USEREVENT):
                self.timer -= 1

        if self.render_mode == "human":
            self._render_frame(curr_gen=curr_gen)

        return observation, self.reward, done, {}, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed: The random seed for the environment (optional).
            options: Additional options for resetting the environment (optional).

        Returns:
            observation: The observation of the environment after the reset.
            info: Additional information about the environment after the reset.
        """
        super().reset(seed=seed)

        if not self.render_mode == "human" and self.repeat_every:
            self.repeat_every.stop()

        self.score = 0
        self.timer = 21  # 20 + 1 because of the timer implementation quirk

        self.player_loc = np.array([WIDTH // 2, HEIGHT // 2])

        self.coin_loc = self.np_random.integers(0, HEIGHT - 20, size=2, dtype=int)

        self.auto_mode = False
        self.done_reset = True
        self.debug = False
        self._counter = 1000
        self.coin_cps = 0
        self.current_step = 0

        self.reward = 0

        if not self.render_mode == "human":
            self.repeat_every = Repeat(1, self._decrease_timer)
            self.repeat_every.start()

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
        """
        Render the frame for the game display.

        Args:
            curr_gen (int): The current generation.

        Returns:
            None
        """
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
                f"reward: {self.reward:.4f}",
            ]

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
