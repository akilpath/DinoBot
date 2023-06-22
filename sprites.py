import pyglet
import random

from timer import Timer


class Player:
    def __init__(self, batch):
        self.width = 50
        #self.sprite = pyglet.shapes.Rectangle(80, 200, self.width, self.width, color=(0, 0, 255, 255), batch=batch)
        self.yspeed = 0
        # acceleration due to gravity
        self.gravity = 2800

        #0 is idle, 1 is jump, 2 is duck
        self.state = 0

        runImages = [pyglet.resource.image("gameAssets/runningdinoA.png"),
                     pyglet.resource.image("gameAssets/runningDinoB.png")]

        duckImages = [pyglet.resource.image("gameAssets/duckDinoA.png"),
                      pyglet.resource.image("gameAssets/duckDinoB.png")]

        idleImage = pyglet.resource.image("gameAssets/idledino.png")
        self.runAnimation = pyglet.image.Animation.from_image_sequence(runImages, duration = 0.1, loop=True)
        self.duckAnimation = pyglet.image.Animation.from_image_sequence(duckImages,duration=0.1,loop=True)
        self.idleImage = idleImage
        self.hitbox = pyglet.shapes.Circle(x=80, y=240, radius=40, color=(0,255,0,255))
        self.sprite = pyglet.sprite.Sprite(img=self.runAnimation, batch = batch)
        self.sprite.x = 80
        self.sprite.y = 200

        self.stateCount = 0

    def updatePos(self, dt):
        newY = int(self.sprite.y + self.yspeed * dt)
        if newY <= 200:
            self.yspeed = 0
            newY = 200
        else:
            self.yspeed -= self.gravity * dt
        self.sprite.y = newY
        self.hitbox.y = self.sprite.y + self.sprite.height // 2
        if self.state == 2:
            self.hitbox.y -= 10
        self.hitbox.x = self.sprite.x + self.sprite.width // 2

    def setState(self, state):
        if self.state == state: return

        if state == 1:
            # jump
            if self.onGround():
                self.yspeed = 1100
                self.sprite.image = self.idleImage
                self.state = state
        elif state == 2:
            # duck
            if self.onGround():
                self.sprite.image = self.duckAnimation
                self.state = state
        else:
            # nothing
            self.sprite.image = self.runAnimation
            self.state = 0
        self.stateCount = 0

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
    def __init__(self):
        self.obstacleType = random.randint(0, 1)
        if self.obstacleType == 0:
            self.height = random.randint(50, 140)
            self.width = random.randint(50, 90)
            self.sprite = pyglet.shapes.Rectangle(1700, 200, self.width, self.height, color=(255, 0, 0, 255))
        else:
            self.width = 70
            self.height = 500
            y = random.randint(260, 350)
            self.height = 500 - y
            self.sprite = pyglet.shapes.Rectangle(1700, y, self.width, self.height, color=(255, 0, 0, 255))

        self.XSPEEDSTART = 600
        self.timer = Timer()
        self.timer.startTimer()

    def update(self, dt, score):
        self.xSpeed = int(self.XSPEEDSTART + score*3)
        self.sprite.x -= int(2*dt*self.xSpeed)

    def x(self):
        return self.sprite.x

    def y(self):
        return self.sprite.y