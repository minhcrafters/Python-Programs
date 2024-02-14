# https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96

import numpy as np
from queue import Queue
import random
import pygame

pygame.init()

SQUARE_SIZE = 5


def create_maze(dim):
    # Create a grid filled with walls
    maze = np.ones((dim * 2 + 1, dim * 2 + 1))

    # Define the starting point
    x, y = (0, 0)
    maze[2 * x + 1, 2 * y + 1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (
                nx >= 0
                and ny >= 0
                and nx < dim
                and ny < dim
                and maze[2 * nx + 1, 2 * ny + 1] == 1
            ):
                maze[2 * nx + 1, 2 * ny + 1] = 0
                maze[2 * x + 1 + dx, 2 * y + 1 + dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze


def find_path(maze):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = (1, 1)
    end = (maze.shape[0] - 2, maze.shape[1] - 2)
    visited = np.zeros_like(maze, dtype=bool)
    visited[start] = True
    queue = Queue()
    queue.put((start, []))
    while not queue.empty():
        (node, path) = queue.get()
        for dx, dy in directions:
            next_node = (node[0] + dx, node[1] + dy)
            if next_node == end:
                return path + [next_node]
            if (
                next_node[0] >= 0
                and next_node[1] >= 0
                and next_node[0] < maze.shape[0]
                and next_node[1] < maze.shape[1]
                and maze[next_node] == 0
                and not visited[next_node]
            ):
                visited[next_node] = True
                queue.put((next_node, path + [next_node]))


def draw_maze(screen, maze):
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            color = (0, 0, 0) if maze[i][j] == 1 else (255, 255, 255)
            pygame.draw.rect(
                screen,
                color,
                (
                    i * SQUARE_SIZE,
                    j * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                ),
            )


def draw_path(path):
    offset = 1
    for coord in path:
        pygame.draw.rect(
            screen,
            "red",
            (
                coord[0] * SQUARE_SIZE + offset,
                coord[1] * SQUARE_SIZE + offset,
                SQUARE_SIZE - offset,
                SQUARE_SIZE - offset,
            ),
        )


if __name__ == "__main__":
    dim = int(input("Enter the dimension of the maze: "))

    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    maze = create_maze(dim)
    path = find_path(maze)

    draw_path_enabled = False
    running = True
    while running:
        screen.fill("black")
        draw_maze(screen, maze)
        if draw_path_enabled:
            draw_path(path)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    draw_path_enabled = not draw_path_enabled
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
