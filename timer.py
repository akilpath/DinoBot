import time


class Timer:

    def __init__(self, endTime=0):
        self.startTime = time.time()
        self.endTime = endTime + time.time()

    def startTimer(self):
        self.startTime = time.time()

    def waitUntil(self, endTime):
        self.endTime = endTime + time.time()
        self.startTimer()

    def getElapsed(self) -> float:
        return time.time() - self.startTime

    def getPassedEnd(self) -> bool:
        return time.time() >= self.endTime
