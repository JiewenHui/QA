class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
        self.allowDotting()
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
