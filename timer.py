import time
class Timer:

    def __init__(self):
        self.startTime = None
        self.endTime = None

    def startTimer(self):
        self.startTime = time.time()

    def getElapsed(self):
        return time.time() - self.startTime
