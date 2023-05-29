import time

import pyglet

class Game(pyglet.window.Window):

    def __init__(self):
        super().__init__()

        self.width = 1500
        self.height = 700
        self.label = pyglet.text.Label('Hello, world',
                                       font_name='Times New Roman',
                                       font_size=36,
                                       x=self.width // 2, y=self.height // 2,
                                       anchor_x='center', anchor_y='center')
        self.floor = pyglet.shapes.Rectangle(0, 0, 1500, 200)
        self.player = Player()

        self.dt = 0
        self.lastFrame = 0

    def run(self):
        self.lastFrame = time.time()
        pyglet.app.run()

    def on_draw(self):
        self.clear()
        self.label.draw()
        self.floor.draw()
        self.player.draw(self.dt)
        thisFrame = time.time()
        self.dt = thisFrame - self.lastFrame
        self.lastFrame = thisFrame

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE and self.player.ground():
            self.player.yvelocity = 1000

    def getState(self):
        pass


class Player:
    def __init__(self):
        self.radius = 30
        self.sprite = pyglet.shapes.Circle(80, 200 + self.radius, self.radius, color=(0, 0, 255, 255))

        self.yvelocity = 0
        #acceleration due to gravity
        self.gravity = 1500

    def draw(self, dt):
        self.updatePos(dt)
        self.sprite.draw()

    def updatePos(self, dt):
        newY = self.sprite.y + self.yvelocity*dt
        if newY <= 200 + self.radius:
            self.yvelocity = 0
            newY = 200 +self.radius
        else:
            self.yvelocity -= self.gravity*dt
        self.sprite.x, self.sprite.y = 80, newY

    def setPos(self, pos):
        self.sprite.x, self.sprite.y = pos

    def ground(self):
        return self.sprite.y == 200 + self.radius

