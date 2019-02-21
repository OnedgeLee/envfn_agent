
class DashBuffer:
    def __init__(self, window_size=10, sliding_size=1, name=None):
        self.window_size = window_size
        self.sliding_size = sliding_size
        self._buffer = []
        self.cur_size = 0
        self.name = name

    def add(self, element):
        if self.cur_size < self.window_size:
            self._buffer.append(element)
            self.cur_size += 1
            
    def get(self):

        if not self.cur_size :
            return []
        elif self.cur_size < self.window_size :
            result = self._buffer[:self.cur_size]
        else: 
            result = self._buffer[:self.window_size]
            self._buffer = self._buffer[self.sliding_size:]
            self.cur_size -= self.sliding_size
        self.get_called = True
        return result



