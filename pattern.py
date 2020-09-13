from entity_tirp import EntityTIRP


def get_str(symbols, relations):
    to_string = ''
    for i in range(0, len(symbols)-1):
        to_string = to_string + str(symbols[i])
        for j in range(i, i + 1):
            to_string = to_string + str(relations[j])

    to_string = to_string + str(symbols[len(symbols)-1])
    return to_string


class Pattern(object):

    def __init__(self, pattern_size=None, symbols=None, relation=None, supporting_instances=None,
                 supporting_entities=None, mean_mean_duration=None, mean_start_offset=None, mean_end_offset=None,
                 vertical_support=None, mean_horizontal_support=None):
        self.__pattern_size: int = pattern_size
        self.__symbols: list = symbols
        self.__relations: list = relation
        self.__supporting_instances: list = supporting_instances
        self.__supporting_entities: list = supporting_entities
        self.__mean_mean_duration: float = mean_mean_duration
        self.__mean_start_offset: float = mean_start_offset
        self.__mean_end_offset: float = mean_end_offset
        self.__vertical_support: int = vertical_support
        self.__mean_horizontal_support: float = mean_horizontal_support

    def get_pattern_size(self) -> int:
        return self.__pattern_size

    def get_rel_size(self) -> int:
        return len(self.__relations)

    def get_symbols(self) -> list:
        return self.__symbols

    def get_relations(self) -> list:
        return self.__relations

    def get_mean_mean_duration(self) -> float:
        return self.__mean_mean_duration

    def get_mean_start_offset(self) -> float:
        return self.__mean_start_offset

    def get_mean_end_offset(self) -> float:
        return self.__mean_end_offset

    def get_mean_horizontal_support(self) -> float:
        return self.__mean_horizontal_support

    def get_vertical_support(self) -> int:
        return self.__vertical_support

    def get_supporting_instances(self) -> list:
        return self.__supporting_instances

    def get_supporting_entity(self, entity_id) -> EntityTIRP:
        entity_tirp: EntityTIRP = self.__supporting_entities[entity_id]
        return entity_tirp

    def __str__(self):
        return get_str(self.__symbols, self.__relations)

    def get_parent_str(self):
        syms = self.__symbols[:-1]
        rel = self.__relations[:-len(syms)]
        return get_str(syms, rel)

    def __eq__(self, other):
        if isinstance(other, Pattern):
            #  I want comparison to be made considering only name and last
            return (self.__symbols, self.__relations) == (other.get_symbols(), other.get_relations())
        else:
            return False
