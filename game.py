import time
from collections import deque

import pyglet
import random
import numpy
from timer import Timer


class Game(pyglet.window.Window):

    def __init__(self):
        super().__init__()
        self.batch = pyglet.graphics.get_default_batch()

        self.width = 1500
        self.height = 700
        self.scoreLbl = pyglet.text.Label('Score',
                                       font_name='Times New Roman',
                                       font_size=36,
                                       x=self.width // 2, y=550,
                                       anchor_x='center', anchor_y='center', batch=self.batch)
        self.floor = pyglet.shapes.Rectangle(0, 0, 1500, 200, batch=self.batch)
        self.player = Player(self.batch)

        self.dt = 0
        self.lastFrame = 0

        self.obstacleSpawnTimer = Timer()
        self.gameTimer = Timer()
        self.obstacles = deque(maxlen=10)
        self.gameEnded = False

        self.gameOverButton = pyglet.shapes.Rectangle(self.width // 2, self.height // 2, 100, 100,
                                                      color=(0, 255, 0, 255))
        self.obstacleBatch = pyglet.graphics.Batch()


    def run(self):
        self.resetGame()
        pyglet.app.run()

    def resetGame(self):
        self.obstacles.clear()
        self.lastFrame = time.time()
        self.obstacleSpawnTimer.waitUntil(random.randint(1, 4))
        self.gameTimer.startTimer()
        self.gameEnded = False

    def on_draw(self):
        if self.gameEnded:
            self.gameOver()
            return

        self.playing()

    def gameOver(self):
        self.gameOverButton.draw()
        pass

    def playing(self):

        self.clear()

        if len(self.obstacles) > 1 and self.obstacles[0].x() < -300:
            self.obstacles.popleft()

        thisFrame = time.time()
        self.dt = thisFrame - self.lastFrame
        self.lastFrame = thisFrame

        if self.obstacleSpawnTimer.getPassedEnd():
            self.obstacles.append(Obstacle(self.obstacleBatch))
            self.obstacleSpawnTimer.waitUntil(random.randint(1, 4))
        self.player.updatePos(self.dt)
        self.batch.draw()

        #self.obstacleBatch.draw()
        for obstacle in self.obstacles:
            obstacle.draw(self.dt)

        self.getState()

        self.gameEnded = self.checkCollisions()

        self.scoreLbl.text = str(int(self.gameTimer.getElapsed()))

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE and self.player.onGround():
            self.player.yspeed = 1100

    def getState(self):
        data = [self.player.y()]

        if len(self.obstacles) >= 2:
            obstacleData = [
                self.obstacles[0].x(),
                self.obstacles[0].height,
                self.obstacles[0].width,
                self.obstacles[1].x(),
                self.obstacles[1].height,
                self.obstacles[1].width,
            ]
        elif len(self.obstacles) == 1:
            obstacleData = [
                self.obstacles[0].x(),
                self.obstacles[0].height,
                self.obstacles[0].width,
                2000,
                0,
                0,
            ]
        else:
            obstacleData = [
                2000,
                0,
                0,
                2000,
                0,
                0,
            ]
        return numpy.array(data + obstacleData)

    def checkCollisions(self):
        vertices = (
            (self.player.x(), self.player.y()),
            (self.player.x() + self.player.width, self.player.y()),
            (self.player.x(), self.player.y() + self.player.width),
            (self.player.x() + self.player.width, self.player.y() + self.player.width))
        for obstacle in self.obstacles:
            for vertex in vertices:
                x, y = vertex
                inXRange = (obstacle.x() <= x <= obstacle.x() + obstacle.width)
                inYRange = (obstacle.y() <= y <= obstacle.y() + obstacle.height)
                if inXRange and inYRange:
                    return True
        return False

    def on_mouse_release(self, x, y, button, modifiers):
        if self.gameEnded:
            if self.gameOverButton.x <= x <= self.gameOverButton.x + self.gameOverButton.width:
                if self.gameOverButton.y <= y <= self.gameOverButton.y + self.gameOverButton.height:
                    self.resetGame()


class Player:
    def __init__(self, batch):
        self.width = 50
        self.sprite = pyglet.shapes.Rectangle(80, 200, self.width, self.width, color=(0, 0, 255, 255), batch = batch)

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
        self.height = random.randint(50, 150)
        self.width = random.randint(50, 100)
        self.sprite = pyglet.shapes.Rectangle(1700, 200, self.width, self.height, color=(255, 0, 0, 255), batch=batch)

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
        self.sprite.x -= int(self.xSpeed * dt)

    def get_pos(self):
        return self.sprite.x, self.sprite.y
