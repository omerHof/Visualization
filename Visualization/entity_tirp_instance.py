class EntityTIRPInstance(object):
    def __init__(self, instance_vec, entity_id, duration, offset_from_start, offset_from_end):
        self.__instance_vec = instance_vec
        self.__entity_id = entity_id
        self.__duration = duration
        self.__offset_from_start = offset_from_start
        self.__offset_from_end = offset_from_end

    def get_instances(self):
        return self.__instance_vec

    def get_entity_id(self):
        return self.__entity_id

    def get_duration(self) -> int:
        return int(self.__duration)

    def get_offset_start(self) -> int:
        return int(self.__offset_from_start)

    def get_offset_end(self):
        return int(self.__offset_from_end)
