class Interval():
    __slots__ = ['start', 'stop']
    def __init__(self, start=float('-Inf'), stop=float('Inf')):
        self.start = start
        self.stop = stop

    def __contains__(self, value):
        return self.start <= value <= self.stop