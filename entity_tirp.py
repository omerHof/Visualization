class EntityTIRP(object):
    def __init__(self, entity_id, start_period, end_period, duration, offset_from_start, offset_from_end):
        self.__entity_id = entity_id
        self.__start_period = start_period
        self.__end_period = end_period
        self.__duration = duration
        self.__offset_from_start = offset_from_start
        self.__offset_from_end = offset_from_end

    def get_entity_id(self):
        return self.__entity_id

    def get_duration(self):
        return self.__duration

    def get_offset_start(self):
        return self.__offset_from_start

    def get_offset_end(self):
        return self.__offset_from_end

    def get_start_period(self):
        return self.__start_period

    def get_end_period(self):
        return self.__end_period
