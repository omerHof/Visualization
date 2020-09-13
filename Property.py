import json

class Property:

    def __init__(self, id):
        self._p_id = id
        self._num_of_bins = 0
        self._maxVal = float('-inf')
        self._minVal = float('inf')
        self._bounds_dic = {}
        self._values = []

    def addValue(self, value):
        self._values += [value]
        if value > self._maxVal:
            self._maxVal = value
        if value < self._minVal:
            self._minVal = value


    def setBinNum(self, num):
        self._num_of_bins = num

    def getID(self):
        return self._p_id

    def getBinNum(self):
        return self._num_of_bins

    def getBoundsDic(self):
        return self._bounds_dic

    def getValues(self):
        return self._values

    def addBounds(self, key, bound):
        if key not in self._bounds_dic.keys():
            self._bounds_dic[key] = [] + bound
        else:
            self._bounds_dic[key] += bound

    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__)


