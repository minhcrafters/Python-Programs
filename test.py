import pygame
import pg_utils

WIDTH, HEIGHT = 1280, 720

pygame.init()
pygame.font.init()
pygame.display.init()

font = pygame.font.Font("./font/MinecraftRegular.otf", 20)
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("shoot the people")

clock = pygame.time.Clock()

text = font.render("Minecraft 2D", False, "white")

running = True
while running:
    screen.fill("black")
    
    exit = False
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                if not exit:
                    exit = True
                
                if exit:
                    pg_utils.draw_text(screen, font, "Are you sure you want to exist?", anchor="center", pos=(WIDTH // 2, HEIGHT // 2))
                    running = False
                
    x, y = pygame.mouse.get_pos()
    
    rect = text.get_rect()
    rect.center = (WIDTH // 2, HEIGHT // 2)

    if rect.collidepoint(x, y):
        text = font.render("Minecraft 2D with mouse lcick ahrwpijedk", False, "white")
    else:
        text = font.render("Minecraft 2D", False, "white")
        
    
    screen.blit(text, rect)
                
    pygame.display.flip()
    clock.tick(60)
    
pygame.quit()