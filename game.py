import time
from collections import deque

import pyglet
import random

from timer import Timer


class Game(pyglet.window.Window):

    def __init__(self):
        super().__init__()

        self.width = 1500
        self.height = 700
        self.label = pyglet.text.Label('Score',
                                       font_name='Times New Roman',
                                       font_size=36,
                                       x=self.width // 2, y=self.height // 2,
                                       anchor_x='center', anchor_y='center')
        self.floor = pyglet.shapes.Rectangle(0, 0, 1500, 200)
        self.player = Player()

        self.dt = 0
        self.lastFrame = 0

        self.obstacles = deque(maxlen=10)

    def run(self):
        self.lastFrame = time.time()
        pyglet.app.run()

    def on_draw(self):
        self.clear()
        self.label.draw()
        self.floor.draw()
        self.player.draw(self.dt)
        for obstacle in self.obstacles:
            obstacle.draw(self.dt)

        if len(self.obstacles) > 1 and self.obstacles[0].x() < -300:
            self.obstacles.popleft()

        thisFrame = time.time()
        self.dt = thisFrame - self.lastFrame
        self.lastFrame = thisFrame

        if random.randint(0, 100) == 1:
            self.obstacles.append(Obstacle())

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE and self.player.ground():
            self.player.yspeed = 1000

    def getState(self):
        pass


class Player:
    def __init__(self):
        self.radius = 30
        self.sprite = pyglet.shapes.Circle(80, 200 + self.radius, self.radius, color=(0, 0, 255, 255))

        self.yspeed = 0
        # acceleration due to gravity
        self.gravity = 2800

    def draw(self, dt):
        self.updatePos(dt)
        self.sprite.draw()

    def updatePos(self, dt):
        newY = self.sprite.y + self.yspeed * dt
        if newY <= 200 + self.radius:
            self.yspeed = 0
            newY = 200 + self.radius
        else:
            self.yspeed -= self.gravity * dt
        self.sprite.x, self.sprite.y = 80, newY

    def x(self):
        return self.sprite.x

    def y(self):
        return self.sprite.y

    def setPos(self, pos):
        self.sprite.x, self.sprite.y = pos

    def ground(self):
        return self.sprite.y == 200 + self.radius


class Obstacle:

    def __init__(self):
        self.height = random.randint(50, 150)
        self.width = random.randint(50, 100)
        self.sprite = pyglet.shapes.Rectangle(1500, 200, self.width, self.height, color=(255, 0, 0, 255))

        self.xSpeed = 500
        self.timer = Timer()
        self.timer.startTimer()

    def draw(self, dt):
        self.sprite.draw()
        self.updatePos(dt)

    def x(self):
        return self.sprite.x

    def y(self):
        return self.sprite.y

    def updatePos(self, dt):
        self.sprite.x -= self.xSpeed * dt

    def get_pos(self):
        return self.sprite.x, self.sprite.y
