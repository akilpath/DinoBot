import time
from collections import deque

import numpy as np
import pyglet
import random
import numpy
import matplotlib.pyplot as plot

from agent import Agent
from timer import Timer
from sprites import Player, Obstacle

import tracemalloc

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
        self.highScoreLbl = pyglet.text.Label("High Score",
                                              font_name='Times New Roman',
                                              font_size=36,
                                              x = 20, y = 630,
                                              anchor_x="left", anchor_y="center", batch=self.batch)
        self.floor = pyglet.shapes.Rectangle(0, 0, 1500, 200, batch=self.batch)
        self.player = Player(self.batch)

        self.dt = 0
        self.lastFrameTime = 0

        self.obstacleSpawnTimer = Timer()
        self.gameTimer = Timer()
        self.obstacles = deque(maxlen=10)
        self.gameEnded = False

        self.gameOverButton = pyglet.shapes.Rectangle(self.width // 2, self.height // 2, 100, 100,
                                                      color=(0, 255, 0, 255))
        self.obstacleBatch = pyglet.graphics.Batch()

        self.score = 0
        self.highScore = 0

        self.agent = Agent()

        self.lastState = None
        self.reward = 0
        self.lastAction = 0

        self.xData = []
        self.yData = []

        self.fig, self.ax = plot.subplots()

        self.MAXEPISODE = 1000
        self.COPYCOUNT = 40

    def run(self):
        self.resetGame()
        pyglet.app.run()

    def resetGame(self):
        self.obstacles.clear()
        self.lastFrameTime = time.time()
        self.obstacleSpawnTimer.waitUntil(random.randint(1, 4))
        self.gameTimer.startTimer()
        self.gameEnded = False
        self.score = 0
        self.lastState = None

    def on_draw(self):
        if self.gameEnded:
            self.gameOver()
            return

        self.playing()

    def gameOver(self):
        if self.score > self.highScore:
            self.highScore = self.score
            self.highScoreLbl.text = f"High Score: {self.highScore:.2f}"

        self.agent.episodeCount += 1
        print(f"Episode: {self.agent.episodeCount}")
        print(f"Agent Epsilon: {self.agent.epsilon}")
        print(f"Score achieved: {self.score:.2f}")
        if self.agent.episodeCount % self.COPYCOUNT == 0:
            self.agent.copyWeights()
            print("Weights copied")

        self.agent.train()
        self.xData.append(self.agent.episodeCount)
        self.yData.append(self.score)
        if self.agent.episodeCount >= self.MAXEPISODE:
            self.end()
            return

        self.agent.decayEpsilon()

        self.resetGame()
        self.gameOverButton.draw()

    def end(self):
        self.ax.plot(self.xData, self.yData)
        plot.savefig("./figures/test4.png")
        pyglet.app.exit()

    def playing(self):

        self.clear()

        if len(self.obstacles) > 1 and self.obstacles[0].x() < -300:
            self.obstacles.popleft()

        thisFrameTime = time.time()
        self.dt = thisFrameTime - self.lastFrameTime
        self.lastFrameTime = thisFrameTime

        if self.obstacleSpawnTimer.getPassedEnd():
            self.obstacles.append(Obstacle(self.obstacleBatch))
            self.obstacleSpawnTimer.waitUntil(random.randint(1, 4))
        self.player.updatePos(self.dt)
        for obstacle in self.obstacles:
            obstacle.update(self.dt, self.score)

        self.batch.draw()
        self.obstacleBatch.draw()

        if self.score > 2.5:
            state = self.getState()
            self.gameEnded = self.checkCollisions()
            if self.gameEnded:
                reward = -10
            else:
                reward = 1

            if self.lastState is not None:
                if reward == -10:
                    for i in range(10):
                        self.agent.saveExperience(self.lastState, self.lastAction, reward, state)
                else:
                    self.agent.saveExperience(self.lastState, self.lastAction, reward, state)


            self.lastAction = int(self.agent.chooseAction(state))
            self.performAction(self.lastAction)
            self.lastState = state

        self.score = self.gameTimer.getElapsed()
        self.scoreLbl.text = f"{self.score:.2f}"

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.jump()

        if symbol == pyglet.window.key.S:
            self.end()


    def jump(self):
        if self.player.onGround():
            self.player.yspeed = 1100

    def performAction(self, action):
        assert isinstance(action, int), "Not int"
        if action == 0:
            self.jump()


    def getState(self):
        data = [self.player.y()]

        if len(self.obstacles) >= 2:
            obstacleData = [
                self.obstacles[0].x(),
                self.obstacles[0].xSpeed,
                self.obstacles[0].width,
                self.obstacles[0].height,
                self.obstacles[1].x(),
                self.obstacles[1].xSpeed,
                self.obstacles[1].width,
                self.obstacles[1].height,
            ]
        elif len(self.obstacles) == 1:
            obstacleData = [
                self.obstacles[0].x(),
                self.obstacles[0].xSpeed,
                self.obstacles[0].width,
                self.obstacles[0].height,
                2000,
                0,
                0,
                0,
            ]
        else:
            obstacleData = [
                2000,
                0,
                0,
                0,
                2000,
                0,
                0,
                0,
            ]
        npData = numpy.array([data + obstacleData])
        return npData

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

