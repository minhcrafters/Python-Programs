import pygame as td
from pygame.math import Vector2
import math

td.init()

screen = td.display.set_mode((800, 600))
td.display.set_caption("lmfao tds")

clock = td.time.Clock()


class Enemy(td.sprite.Sprite):
    def __init__(self, waypoints, image):
        td.sprite.Sprite.__init__(self)
        self.waypoints = waypoints
        self.pos = Vector2(self.waypoints[0])
        self.target_waypoint = 1
        self.speed = 2
        self.angle = 0
        self.original_image = image
        self.image = td.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def move(self):
        # define a target waypoint
        if self.target_waypoint < len(self.waypoints):
            self.target = Vector2(self.waypoints[self.target_waypoint])
            self.movement = self.target - self.pos
        else:
            # enemy has reached the end of the path
            self.kill()

        # calculate distance to target
        dist = self.movement.length()
        print(dist)
        # check if remaining distance is greater than the enemy speed
        if dist >= self.speed:
            self.pos += self.movement.normalize() * self.speed
        else:
            if dist != 0:
                self.pos += self.movement.normalize() * dist
            self.target_waypoint += 1

    def rotate(self):
        # calculate distance to next waypoint
        dist = self.target - self.pos
        # use distance to calculate angle
        self.angle = math.degrees(math.atan2(dist[1], dist[0]))
        # rotate image and update rectangle
        self.image = td.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def update(self):
        self.move()
        self.rotate()

image = td.image.load("./img/zombie.png").convert_alpha()
image = td.transform.scale(image, (image.get_width() * 0.5, image.get_height() * 0.5))
image = td.transform.rotate(image, 180)

enemy = Enemy([[10, 100], [20, 500], [60, 100], [50, 200]], image)

group = td.sprite.Group()

group.add(enemy)

running = True
while running:
    screen.fill("black")
    for event in td.event.get():
        if event.type == td.QUIT:
            running = False
    
    enemy.update()
    
    group.draw(screen)
    
    td.display.flip()
    clock.tick(60)
