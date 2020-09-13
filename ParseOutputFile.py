import os.path
import sys
import json
import re
from TIRP import TIRP
from SymbolicTimeInterval import SymbolicTimeInterval
from SupportingInstance import SupportingInstance
rel_allen_seven = 7


def get_supporting_instances(entities, instances, line_vector, symbols, index, next_line):
    next_line_parsed = re.split('[ \[ , \]]', next_line)
    line = line_vector[index+8:]
    for word in range(0, len(line) - 4, 5):
        entity_id = line[word]
        instance_vec = []
        symbolic_list = []
        tis = list(filter(None, line[word + 1].split(']')))
        for t in range(0, len(tis)):
            times = tis[t].split('-')
            start_time = int(times[0].replace("[", ""))
            end_time = int(times[1])
            if start_time == end_time:
                sys.exit("Error! Start time can't be equal to end time! please change KLC code")
            symbolic = SymbolicTimeInterval(start_time=start_time, end_time=end_time, symbol=symbols[t])#, duration=tis[t+1], offset_from_start=tis[t+2], offset_from_end=tis[t+3])
            symbolic_list.append(symbolic)
        instance_vec.append(symbolic_list)
        if entity_id in entities:
            instances[len(instances) - 1].add_list_to_intervals(instance_vec)
        else:
            if len(instances) > 0:
                instances[len(instances) - 1].set_means()
            for i in range(0, len(next_line_parsed)-9, 11):
                if entity_id == next_line_parsed[i]:
                    mean_duration = float(next_line_parsed[i+7])
                    mean_offset_from_start = float(next_line_parsed[i+8])
                    mean_offset_from_end = float(next_line_parsed[i+9])
                    break
            support_instance = SupportingInstance(entityId=str(entity_id), symbolic_intervals=instance_vec,
                                                  mean_duration=mean_duration, mean_offset_from_start=mean_offset_from_start,mean_offset_from_end=mean_offset_from_end)
            instances.append(support_instance)
            entities.append(entity_id)
    instance_vec.clear()
    # for instance in instances:
    #     for i in range(0, symbols+1):
    #         mean_of_each_interval[i] += instance[0][i]
    # for i in range(0, len(mean_of_each_interval)):
    #         mean_of_each_interval[i] = mean_of_each_interval[i] / len(entities)
    # support_instance.set___mean_of_each_interval(mean_of_each_interval)


def input_validation(filename):
    if not os.path.isfile(filename):
        print("Wrong file path to parse, please fix the path and try again")
        return False
    return True


def parse_output_file(filename, rel_number, states, path, class_name, min_ver_support, class_1_tirp_file_name, to_add_entities):
    """
    This function create TIRP list from KarmaLego output file.
    Output file structure: [0]TIRP_size [1]symbolNumber-symbolNumber-sym...- [2]rel.rel.rel... [3]mean_duration
    [4]mean_offset_from_start [5]mean_offset_from_end  [6]vertical_support
    [7]mean_horizontal_support [8]entity_id [9][start_time-end_time][10] duration [11]offset_from_start [12]offset_from_end
    :param filename:
    :param rel_number:
    :return: TIRPs list
    """
    if not input_validation(filename):
        return

    if rel_allen_seven is rel_number:
        relations_dict = {"<": "before", "m": "meets", "o": "overlaps", "f": "finished by", "c": "contains",
                          "=": "equals", "s": "starts", "-": 7}
    else:
        print("Wrong number of relations")
        return

    TIRP_list = []
    lines = [line.rstrip('\n') for line in open(filename)]
    for i in range(0, len(lines)-1):
        if i % 2 == 1:
            continue
        line_vector = lines[i].split()
        next_line = lines[i+1]
        instances = []
        entities = list()
        TIRP_size = int(line_vector[0])
        symbols = list(filter(None, line_vector[1].split('-')))
        if states:
            for i in range(0, len(symbols)):
                symbol = states[symbols[i]]
                symbols[i] = symbol
        if TIRP_size >1:
            index = 0;
            relations = list(filter(None, line_vector[index+2].split('.')))
            for r in range(0, len(relations)):
                relations[r] = relations_dict[relations[r]]
        else:
            relations = list()
            index = -1;
        mean_duration = float(line_vector[index+3])
        mean_offset_from_start = float(line_vector[index+4])
        mean_offset_from_end = float(line_vector[index+5])
        vertical_support = int(line_vector[index+6])
        mean_horizontal_support = float(line_vector[index+7])
        get_supporting_instances(entities, instances, line_vector, symbols, index=index, next_line=next_line)
        TIRP_obj = TIRP(tirp_size=TIRP_size, symbols=symbols, relation=relations, supporting_instances=instances,
                        supporting_entities=entities, vertical_support=vertical_support,
                        mean_horizontal_support=mean_horizontal_support, mean_duration=mean_duration,
                        mean_offset_from_start=mean_offset_from_start, mean_offset_from_end=mean_offset_from_end,
                        path=path, min_vertical_support=min_ver_support)
        if class_name == 'class_0':
            class_1_tirp = find_tirp_in_class_1(path, TIRP_obj, class_1_tirp_file_name, to_add_entities)
            TIRP_obj.set_exist_in_class_0()
            if class_1_tirp:
                if not to_add_entities:
                    class_1_tirp = find_tirp_in_class_1(path, TIRP_obj, class_1_tirp_file_name, True)
                TIRP_obj.set_class_1_properties(class_1_tirp)
        if not to_add_entities:
            TIRP_obj.set_supporting_instances(list())
            TIRP_obj.set_supporting_entitie(list())
        TIRP_list.append(TIRP_obj)
    return TIRP_list


def parse_TIRP(line1, line2, states, path, class_name, min_ver_support, class_1_tirp_file_name, second_class_output_file_name, to_add_entities):
    relations_dict = {"<": "before", "m": "meets", "o": "overlaps", "f": "finished by", "c": "contains", "=": "equals", "s": "starts", "-": 7}
    line_vector = line1.split()
    instances = []
    entities = list()
    TIRP_size = int(line_vector[0])
    symbols = list(filter(None, line_vector[1].split('-')))
    if states:
        for i in range(0, len(symbols)):
            symbol = states[symbols[i]]
            symbols[i] = symbol
    if TIRP_size >1:
        index = 0;
        relations = list(filter(None, line_vector[index+2].split('.')))
        for r in range(0, len(relations)):
            relations[r] = relations_dict[relations[r]]
    else:
        relations = list()
        index = -1;
    mean_duration = float(line_vector[index+3])
    mean_offset_from_start = float(line_vector[index+4])
    mean_offset_from_end = float(line_vector[index+5])
    vertical_support = int(line_vector[index+6])
    mean_horizontal_support = float(line_vector[index+7])
    get_supporting_instances(entities=entities, instances=instances, line_vector=line_vector, symbols=symbols,
                             index=index, next_line=line2)
    TIRP_obj = TIRP(tirp_size=TIRP_size, symbols=symbols, relation=relations, supporting_instances=instances,
                    supporting_entities=entities, vertical_support=vertical_support,
                    mean_horizontal_support=mean_horizontal_support,mean_duration=mean_duration,
                    mean_offset_from_start=mean_offset_from_start, mean_offset_from_end=mean_offset_from_end, path=path,
                    min_vertical_support=min_ver_support)
    if class_name == 'class_0' and second_class_output_file_name != 'File does not exist':
        TIRP_obj.set_exist_in_class_0()
        class_1_tirp = find_tirp_in_class_1(path, TIRP_obj, class_1_tirp_file_name, to_add_entities)
        if class_1_tirp:
            if not to_add_entities:
                class_1_tirp = find_tirp_in_class_1(path, TIRP_obj, class_1_tirp_file_name, True)
            TIRP_obj.set_class_1_properties(class_1_tirp)
    if not to_add_entities:
        TIRP_obj.set_supporting_instances(list())
        TIRP_obj.set_supporting_entitie(list())
    return TIRP_obj


def find_tirp_in_class_1(path, class_0_tirp, class_1_tirp_file_name, to_add_entities):
    if to_add_entities:
        dir_path = path + '/tempChunks1_with_entities/'
    else:
        dir_path = path + '/tempChunks1/'
    path_to_class_1_tirp = dir_path + class_1_tirp_file_name
    if os.path.isfile(path_to_class_1_tirp):
        with open(path_to_class_1_tirp, "r") as fr:
            tirp_dict = json.load(fr)
            class_1_tirp = TIRP()
            class_1_tirp.__dict__.clear()
            class_1_tirp.__dict__.update(tirp_dict)
            class_0_tirp_size = class_0_tirp.get_tirp_size()
            class_1_tirp_size = class_1_tirp.get_tirp_size()
            found = True
            if class_0_tirp_size == 1: #root element
                class_1_tirp.set_exist_in_class_0()
                with open(path_to_class_1_tirp, "w") as fw:
                    class_1_tirp_json = json.dumps(class_1_tirp, default=lambda x: x.__dict__)
                    fw.write(class_1_tirp_json)
                root_elements_class_1 = list()
                with open(dir_path + 'root.txt', "r") as fr:
                    lines = fr.readlines()
                    for line in lines:
                        tirp_obj_class_1 = TIRP()
                        tirp_obj_class_1.__dict__.clear()
                        tirp_obj_class_1.__dict__.update(json.loads(line))
                        root_elements_class_1.append(tirp_obj_class_1)
                for root_element in root_elements_class_1:
                    if root_element.get_symbols()[0] == class_1_tirp.get_symbols()[0]:
                        root_element.set_exist_in_class_0()
                        break
                os.remove(dir_path + 'root.txt')
                with open(dir_path + 'root.txt', "a") as fr:
                    for root_element in root_elements_class_1:
                        r = json.dumps(root_element, default=lambda x: x.__dict__)
                        fr.write("%s\n" % r)
                return class_1_tirp
            else:
                father = class_1_tirp
                childs = class_1_tirp.get_childes()
                while class_1_tirp_size < class_0_tirp_size and len(childs) > 0 and found:
                    found = False
                    for index, child in enumerate(childs):
                        curr_tirp = TIRP()
                        curr_tirp.__dict__.clear()
                        curr_tirp.__dict__.update(child)
                        curr_size = curr_tirp.get_tirp_size()
                        new_symbols = class_0_tirp.get_symbols()[:curr_size]
                        num_of_new_rels = int(curr_size * (curr_size - 1) / 2)
                        new_rels = class_0_tirp.get_rels()[:num_of_new_rels]
                        if curr_tirp.get_symbols() == new_symbols:
                            if curr_tirp.get_rels() == new_rels:
                                if curr_tirp.get_tirp_size() == class_0_tirp.get_tirp_size():
                                    curr_tirp.set_exist_in_class_0()
                                    childs[index] = curr_tirp
                                    father.update_childs(childs)
                                    with open(path_to_class_1_tirp, "w") as fw:
                                        class_1_tirp_json = json.dumps(class_1_tirp, default=lambda x: x.__dict__)
                                        fw.write(class_1_tirp_json)
                                    return curr_tirp
                                else:
                                    father = curr_tirp
                                    childs = curr_tirp.get_childes()
                                    class_1_tirp_size = curr_tirp.get_tirp_size()
                                    found = True
                                    break
                return None
    else:
        if class_0_tirp.get_tirp_size() == 1:
            root_elements_class_1 = list()
            with open(dir_path + 'root.txt', "r") as fr:
                lines = fr.readlines()
                for line in lines:
                    tirp_obj_class_1 = TIRP()
                    tirp_obj_class_1.__dict__.clear()
                    tirp_obj_class_1.__dict__.update(json.loads(line))
                    root_elements_class_1.append(tirp_obj_class_1)
            for root_element in root_elements_class_1:
                if root_element.get_symbols()[0] == class_0_tirp.get_symbols()[0]:
                    root_element.set_exist_in_class_0()
                    os.remove(dir_path + 'root.txt')
                    with open(dir_path + 'root.txt', "a") as fr:
                        for element in root_elements_class_1:
                            r = json.dumps(element, default=lambda x: x.__dict__)
                            fr.write("%s\n" % r)
                    return root_element
            return None
        else:
            return None


def parse_states_file(path):
    """
    This function parses the state file and init the states dictionary. The file must contain the columns - StateID, TemporalPropertyName, BinLabel
    :return:
    """
    states = dict()
    states_by_name = dict()
    states_from_file = []
    with open(path, 'r') as fs:
        lines = fs.readlines()
        for line in lines:
            states_from_file.append(json.loads(line))
    # counter = 1
    # curr_id = ''
    for i in range(0, len(states_from_file)):
        state_id = states_from_file[i]['StateID']
        if 'TemporalPropertyName' in states_from_file[i].keys():
            first_part = states_from_file[i]['TemporalPropertyName'].rstrip()
        else:
            first_part = states_from_file[i]['TemporalPropertyID']
        if 'BinLabel' in states_from_file[i].keys():
            second_part = states_from_file[i]['BinLabel'].rstrip()
        else:
            second_part = states_from_file[i]['BinID'].rstrip()
            # if curr_id == states_from_file[i]['TemporalPropertyID']:
            #     second_part = counter
            #     counter += 1
            # else:
            #     counter = 1
            #     second_part = counter
            #     curr_id = states_from_file[i]['TemporalPropertyID']
        state_name = first_part + '.' + second_part
        states[state_id] = state_name
        states_by_name[state_name] = state_id
    return states, states_by_name


# parse_states_file()
# parse_output_file('./RawDataWithOffset.txt', 7)


