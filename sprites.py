import pyglet
import random

from timer import Timer


class Player:
    def __init__(self, batch):
        self.width = 50
        self.sprite = pyglet.shapes.Rectangle(80, 200, self.width, self.width, color=(0, 0, 255, 255), batch=batch)

        self.yspeed = 0
        # acceleration due to gravity
        self.gravity = 2800

    def updatePos(self, dt):
        newY = int(self.sprite.y + self.yspeed * dt)
        if newY <= 200:
            self.yspeed = 0
            newY = 200
        else:
            self.yspeed -= self.gravity * dt
        self.sprite.y = newY

    def x(self):
        return self.sprite.x

    def y(self):
        return self.sprite.y

    def height(self):
        return self.sprite.height

    def setPos(self, pos):
        self.sprite.x, self.sprite.y = pos

    def onGround(self):
        return self.sprite.y == 200


class Obstacle:

    def __init__(self, batch):
        self.height = random.randint(50, 140)
        self.width = random.randint(50, 90)
        self.sprite = pyglet.shapes.Rectangle(1700, 200, self.width, self.height, color=(255, 0, 0, 255), batch=batch)

        self.xSpeed = 500
        self.timer = Timer()
        self.timer.startTimer()

    def update(self, dt, score):
        self.xSpeed = int(500 + score)
        self.sprite.x -= int(dt*self.xSpeed)

    def x(self):
        return self.sprite.x

    def y(self):
        return self.sprite.y

    def get_pos(self):
        return self.sprite.x, self.sprite.y