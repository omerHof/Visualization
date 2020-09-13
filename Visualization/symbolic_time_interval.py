class SymbolicTimeInterval(object):
    """
     Symbolic time Interval representing a time interval with start time and end time and corresponds to a given symbol.
    """

    def __init__(self, start_time, end_time, symbol, var_id=None):
        self.start_time = start_time
        self.end_time = end_time
        self.symbol = symbol
        self.var_id = var_id

    def __eq__(self, other) -> bool:
        return self.start_time == other.get_start_time() and self.end_time == other.get_end_time() and \
               self.symbol == other.get_symbol()

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_symbol(self):
        return self.symbol

    def get_var_id(self):
        return self.var_id

    def get_duration(self):
        return self.end_time - self.start_time

    def get_gap_time(self, other) -> int:
        """
        Get the gap between two symbolic time intervals
        ------ self ------
                            !GAP!
                                   ----- other -----
        :param other:
        :return:
        """
        if other is None:
            raise Exception("STI equals to None")
        elif other.get_start_time() < self.get_start_time() or other.get_start_time() < self.get_end_time():
            raise Exception("Other can't be after the current sti")

        gap = other.get_start_time() - self.get_end_time()
        return gap

    def to_string(self):
        return 'SymbolicTimeInterval: { Symbol: '+str(self.symbol)+', startTime: '+str(self.start_time)+\
               ', endTime: '+str(self.end_time)+', varID: '+str(self.var_id)+'}'


    def copy(self):
        """
        create new SymbolicTimeInterval and copy all current variables
        :return: SymbolicTimeInterval,copy of this tirp
        """
        new_SymbolicTimeInterval = SymbolicTimeInterval()
        new_SymbolicTimeInterval._start_time = self.start_time
        new_SymbolicTimeInterval._end_time=self.end_time
        new_SymbolicTimeInterval._symbol = self.symbol
        new_SymbolicTimeInterval._var_id=self.var_id
        return new_SymbolicTimeInterval
