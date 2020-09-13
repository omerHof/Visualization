# import os.path
# import time
# from typing import Dict
#
# from entity_tirp import EntityTIRP
# from entity_tirp_instance import EntityTIRPInstance
# from pattern import Pattern
# from symbolic_time_interval import SymbolicTimeInterval
# import ParseOutputFile
#
# rel_allen_seven = 7
# """
# This function create TIRP list from KarmaLego output file.
# Output file structure:
# [0]tirp_size
# [1]symbolNumber-symbolNumber-sym...-
# [2]rel.rel.rel...
# [3]Mean_Mean_Duration
# [4]Mean_Start_Offset
# [5]Mean_End_Offset
# [6]Vertical_Support
# [7]Mean_Horizontal_Support
# [8]entity_id
# [9][start_time-end_time]
# [10]Duration
# [11]Offset from START
# [12]Offset from END
# [13]entity_id
# [14][start_time-end_time]...
# For example:
# INDEX: [0] | [1]     | [2] | [3] | [4] | [5]  | [6] | [7] | [8] |         [9]        | [10] | [11] | [12]   ...
# VALUES: 2  | 12-9-   |  <. | 35.0| 5.0 | 60.0 | 10  | 1.0 | 71  | [605-610][620-640] | 35   |  5   |  60    ...
# """
# locations: Dict = {
#     'loc_tirp_size': 0,
#     'loc_symbols': 1,
#     'loc_relations': 2,
#     'loc_mean_mean_duration': 3,
#     'loc_mean_start_offset': 4,
#     'loc_mean_end_offset': 5,
#     'loc_vertical_support': 6,
#     'loc_mean_horizontal_support': 7,
#     'loc_start_entities': 8
# }
#
#
# def get_supporting_instances(line_vector, symbols) -> list:
#     entity_instances = []
#     steps_each_iteration = 5
#     for instance_index in range(locations['loc_start_entities'], len(line_vector) - 1, steps_each_iteration):
#         start_entity_index = instance_index
#         entity_id = int(line_vector[instance_index])
#         start_entity_index += 1
#         instance_vec = []
#         start_time_index, end_time_index = 0, 1
#         time_intervals = list(filter(None, line_vector[start_entity_index].split(']')))
#         for interval_index in range(0, len(time_intervals)):
#             interval: str = time_intervals[interval_index].split('-')
#             start_time = int(interval[start_time_index].replace("[", ""))
#             end_time = int(interval[end_time_index])
#             if start_time == end_time:
#                 raise Exception("Error! Start time can\'t be equal to end time!")
#             instance_vec.append(SymbolicTimeInterval(start_time=start_time,
#                                                      end_time=end_time,
#                                                      symbol=symbols[interval_index]))
#         start_entity_index += 1
#         duration = line_vector[start_entity_index]
#         start_entity_index += 1
#         offset_from_start = line_vector[start_entity_index]
#         start_entity_index += 1
#         offset_from_end = line_vector[start_entity_index]
#         entity_instances.append(EntityTIRPInstance(instance_vec=instance_vec,
#                                                    entity_id=entity_id,
#                                                    duration=duration,
#                                                    offset_from_start=offset_from_start,
#                                                    offset_from_end=offset_from_end))
#     return entity_instances
#
#
# def input_validation(filename):
#     if not os.path.isfile(filename):
#         raise Exception("You gave our algorithm a wrong input path of patterns file: {},"
#                         " Please fix this path and try again".format(filename))
#     return True
#
#
# def get_supporting_entities(line_vector) -> Dict:
#     entities_list = {}
#     steps_each_iteration = 6
#     for instance_index in range(0, len(line_vector) - 1, steps_each_iteration):
#         start_entity_index = instance_index
#         entity_id = int(line_vector[start_entity_index].replace(",", ""))
#         start_entity_index += 1
#         start_period = int(line_vector[start_entity_index].replace(",", "").replace("[", ""))
#         start_entity_index += 1
#         end_period = int(line_vector[start_entity_index].replace("]", "").replace(":", ""))
#         start_entity_index += 1
#         mean_durations = float(line_vector[start_entity_index].replace("]", ""))
#         start_entity_index += 1
#         offset_from_start = float(line_vector[start_entity_index].replace("]", ""))
#         start_entity_index += 1
#         offset_from_end = float(line_vector[start_entity_index].replace(",", ""))
#         entities_list[entity_id] = EntityTIRP(entity_id=entity_id,
#                                               start_period=start_period,
#                                               end_period=end_period,
#                                               duration=mean_durations,
#                                               offset_from_start=offset_from_start,
#                                               offset_from_end=offset_from_end)
#     return entities_list
#
#
# def get_tirps(file_path):
#     tirp_list = []
#     lines = [line.rstrip('\n') for line in open(file_path)]
#     lines_size = len(lines)
#     for line_index in range(0, lines_size, 2):
#         first_line = lines[line_index]
#         first_line_vector = first_line.split()
#         second_line = lines[line_index + 1]
#         second_line_vector = second_line.split()
#         tirp_size = int(first_line_vector[locations['loc_tirp_size']])
#         symbols = list(filter(None, first_line_vector[locations['loc_symbols']].split('-')))
#         relations = list(filter(None, first_line_vector[locations['loc_relations']].split('.')))
#         mean_mean_duration = float(first_line_vector[locations['loc_mean_mean_duration']])
#         mean_start_offset = float(first_line_vector[locations['loc_mean_start_offset']])
#         mean_end_offset = float(first_line_vector[locations['loc_mean_end_offset']])
#         vertical_support = int(first_line_vector[locations['loc_vertical_support']])
#         mean_horizontal_support = float(first_line_vector[locations['loc_mean_horizontal_support']])
#         instances: list = get_supporting_instances(line_vector=first_line_vector,
#                                                    symbols=symbols)
#         entities: Dict = get_supporting_entities(line_vector=second_line_vector)
#         tirp_obj = Pattern(pattern_size=tirp_size,
#                            symbols=symbols,
#                            relation=relations,
#                            supporting_instances=instances,
#                            supporting_entities=entities,
#                            mean_mean_duration=mean_mean_duration,
#                            mean_start_offset=mean_start_offset,
#                            mean_end_offset=mean_end_offset,
#                            vertical_support=vertical_support,
#                            mean_horizontal_support=mean_horizontal_support)
#         tirp_list.append(tirp_obj)
#     return tirp_list
#
#
#
# start = time.time()
# tirp_list_class_zero = get_tirps("./DataSets/out3/chunks/1.txt")
# end = time.time()
# print(end - start)
# start = time.time()
# states = ParseOutputFile.parse_states_file("./DataSets/out3/states.json")
# start = time.time()
# tirp_list_class_zero = ParseOutputFile.parse_output_file("./DataSets/out3/chunks/1.txt",7,states,None)
# end = time.time()
# print(end - start)
# print('')
#     # tirp_list_class_one = get_tirps(one_class_filename)
#     #
#     # # Feature selection
#     # tirp_list_class_zero_and_one = [item for item in tirp_list_class_one if item in tirp_list_class_zero]
#     # tirp_list_only_in_one = [item for item in tirp_list_class_one if item not in tirp_list_class_zero]
#     # return tirp_list_class_one, tirp_list_class_zero_and_one
