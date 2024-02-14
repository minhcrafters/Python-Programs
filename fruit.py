import sys
import pygame
import random
import pyautogui

score = 0

WIDTH = HEIGHT = 600

pygame.init()
pygame.font.init()  # Initialize the font module

font = pygame.font.SysFont("Consolas", 18)  # Or any other preferred font
# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=1)
pygame.display.set_caption("Shoot the Fruit")

image = pygame.image.load("./img/apple.png").convert_alpha()

clock = pygame.time.Clock()

apple_x = random.randint(0, WIDTH // 2)
apple_y = random.randint(0, HEIGHT // 2)

image_rect = image.get_rect()


def draw_scores():
    score_text = font.render(str(score), True, (255, 255, 255))
    text_x = (WIDTH - score_text.get_width()) // 2
    text_y = (HEIGHT - font.get_height() // 2) - 20
    screen.blit(score_text, (text_x, text_y))


def main():
    global score, apple_x, apple_y, image_rect, image

    running = True
    autoclick = False
    while running:
        screen.fill("black")

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    autoclick = not autoclick
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if image_rect.collidepoint(x, y):
                    score += 1
                    apple_x = random.uniform(0, random.uniform(apple_x, WIDTH))
                    apple_y = random.uniform(0, random.uniform(apple_y, HEIGHT))
                    if (
                        image_rect.center[0] <= 10 or image_rect.center[0] >= WIDTH - 20
                    ) or (
                        image_rect.center[1] <= 10
                        or image_rect.center[1] >= HEIGHT - 20
                    ):
                        apple_x = random.uniform(0, WIDTH - apple_x - 100)
                        apple_y = random.uniform(0, HEIGHT - apple_y - 100)

        image_rect.x = apple_x
        image_rect.y = apple_y

        screen.blit(image, image_rect)

        draw_scores()

        # Auto-click lmao
        if autoclick:
            if pygame.mouse.get_focused():
                pygame.mouse.set_pos(image_rect.center)
                pyautogui.click()

        

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
