import ParseOutputFile
import json
import os
import copy
from TIRP import TIRP

TIRP_per_file = dict()


# def init_data_set(data_set_name, KL_outputfile, states_file, entities_file):
#     path = 'DataSets/' + data_set_name
#     if not os.path.exists(path):
#         os.mkdir(path)
#
#         shutil.copy2(path + '/'+KL_outputfile, path)
#         shutil.copy2(path + '/' + states_file, path)
#         shutil.copy2(path + '/' + entities_file, path)
#         #PreProccesing.create_index_files(KL_outputfile,)

def parse_main_index(path, dir_path_class_0, dir_path_class_1, states, states_by_name, class_name, min_ver_support, second_class_output_file_name, to_add_entities):
    TIRP_per_file.clear()
    root_elements = list()
    if class_name == 'class_0':
        path_to_chuncks = path + '/chunks/'
    else:
        path_to_chuncks = path + '/chunks1/'
    with open(path_to_chuncks + 'main_Index.txt') as fp:
        line1 = fp.readline()
        while line1:
            line2 = fp.readline()
            file_name = fp.readline().rstrip()
            # ParseOutputFile.parse_states_file()
            tirp = ParseOutputFile.parse_TIRP(line1, line2, states, path, class_name, min_ver_support, file_name, second_class_output_file_name, to_add_entities)
            if os.path.isfile(path_to_chuncks + file_name):
                tirp.set_childes(has_childs=True)
                tirp_with_childs = copy.deepcopy(tirp)
                tirp_with_childs = get_sub_tree(tirp_with_childs, states, states_by_name, path, class_name, min_ver_support, file_name, to_add_entities)
                tirp_with_childs_json = json.dumps(tirp_with_childs, default=lambda x: x.__dict__)
                if class_name == 'class_0':
                    path_to_file = path + dir_path_class_0 + file_name
                else:
                    path_to_file = path + dir_path_class_1 + file_name
                with open(path_to_file, "w") as fs:
                    fs.write(tirp_with_childs_json)
            else:
                if class_name == 'class_0' and second_class_output_file_name != 'File does not exist':
                    if not to_add_entities:  # take the tirp with the entities
                        dir_path_class_1_new = '/tempChunks1_with_entities/'
                    else:
                        dir_path_class_1_new = dir_path_class_1
                    if os.path.isfile(path + dir_path_class_1_new + file_name):
                        with open(path + dir_path_class_1_new + file_name, "r") as fr:
                            tirp_dict = json.load(fr)
                            tirp_obj = TIRP()
                            tirp_obj.__dict__.clear()
                            tirp_obj.__dict__.update(tirp_dict)
                            tirp.set_class_1_properties(tirp_obj)
                            if not to_add_entities:
                                tirp.set_supporting_instances(list())
                                tirp.set_supporting_entitie(list())
                    else:
                        with open(path + dir_path_class_1_new + 'root.txt', "r") as fr:
                            lines = fr.readlines()
                            for line in lines:
                                tirp_obj_class_1 = TIRP()
                                tirp_obj_class_1.__dict__.clear()
                                tirp_obj_class_1.__dict__.update(json.loads(line))
                                if tirp_obj_class_1.get_symbols() == tirp.get_symbols():
                                    if tirp_obj_class_1.get_rels == tirp.get_rels():
                                        tirp.set_class_1_properties(tirp_obj_class_1)
                                        if not to_add_entities:
                                            tirp.set_supporting_instances(list())
                                            tirp.set_supporting_entitie(list())
                                        break
            s = json.dumps(tirp, default=lambda x: x.__dict__)
            root_elements.append(s)
            # root_elements.append(TIRP)
            TIRP_name = tirp.get_unique_name()
            TIRP_per_file[TIRP_name] = file_name
            line1 = fp.readline()
        return root_elements


def get_sub_tree(TIRP, states, states_by_name, path, class_name, min_ver_support, class_1_tirp_file_name, to_add_entities):
    if states_by_name:
        TIRP_name = states_by_name[TIRP.get_symbols()[0]] + '.txt'
    else:
        TIRP_name = TIRP.get_symbols()[0] + '.txt'
    if class_name == 'class_0':
        file_name = path + '/chunks' + '/' + TIRP_name
    else:
        file_name = path + '/chunks1' + '/' + TIRP_name
    # ParseOutputFile.parse_states_file()
    TIRPs = ParseOutputFile.parse_output_file(file_name, 7, states, path, class_name, min_ver_support, class_1_tirp_file_name, to_add_entities)
    TIRP.set_childes(TIRPs_in_output_file=TIRPs)
    return TIRP


def find_Path_of_tirps(symbols, rels, data_set_path, states,states_by_name=None, to_add_entities=None):
    try:
        relations_dict = {"<": "before", "m": "meets", "o": "overlaps", "f": "finished by", "c": "contains", "=": "equals",
                          "s": "starts","-": 7}
        rels = list(filter(None, rels.split('.')))
        tirps_path = []
        symbols = list(filter(None, symbols.split('-')))
        if not to_add_entities:
            file_name = symbols[0] + '.txt'
            dir_path = '/tempChunks/'
            for r in range(0, len(rels)):
                rels[r] = relations_dict[rels[r]]
            for i in range(0, len(symbols)):
                symbol = states[symbols[i]]
                symbols[i] = symbol
        else:
            file_name = states_by_name[symbols[0]] + '.txt'
            dir_path = '/tempChunks_with_entities/'
        tirp_size = len(symbols)
        if os.path.isfile(data_set_path + dir_path + file_name):
            with open(data_set_path + dir_path + file_name, "r") as fr:
                tirp_dict = json.load(fr)
                tirp_obj = TIRP()
                tirp_obj.__dict__.clear()
                tirp_obj.__dict__.update(tirp_dict)
                tirps_path.append(tirp_obj)
                if tirp_size > 1:
                    childs = tirp_obj.get_childes()
                    while len(tirps_path) < tirp_size:
                        for child in childs:
                            curr_tirp = TIRP()
                            curr_tirp.__dict__.clear()
                            curr_tirp.__dict__.update(child)
                            curr_size = curr_tirp.get_tirp_size()
                            new_symbols = symbols[:curr_size]
                            num_of_new_rels = int(curr_size * (curr_size - 1) / 2)
                            new_rels = rels[:num_of_new_rels]
                            if curr_tirp.get_symbols() == new_symbols:
                                if curr_tirp.get_rels() == new_rels:
                                    tirps_path.append(curr_tirp)
                                    childs = curr_tirp.get_childes()
                                    break
                return tirps_path
        else:
            with open(data_set_path + dir_path + 'root.txt', "r") as fr:
                roots_from_file = fr.readlines()
                for line in roots_from_file:
                    tirp_dict = json.loads(line)
                    tirp_obj = TIRP()
                    tirp_obj.__dict__.clear()
                    tirp_obj.__dict__.update(tirp_dict)
                    if tirp_obj.get_symbols() == symbols:
                        if tirp_obj.get_rels() == rels:
                            return [tirp_obj]

    except Exception as e:
        print(e)


def marge_trees(data_set_path, dir_path_class_0 , dir_path_class_1, states_by_name, min_ver_support):
    """
    This function find tirps that exist only in class 1 and insert them to the right place in the tree
    :param data_set_path:
    :param states_by_name:
    :param min_ver_support:
    :return:
    """
    root_elements_class_1 = list()
    root_elements_class_0 = list()
    temp_list = list() # for tirps in class 1 that not have childs
    with open(data_set_path + dir_path_class_0 + 'root.txt', "r") as fr:  #load roots from class 0
        lines = fr.readlines()
        for line in lines:
            tirp_obj_class_0 = TIRP()
            tirp_obj_class_0.__dict__.clear()
            tirp_obj_class_0.__dict__.update(json.loads(line))
            root_elements_class_0.append(tirp_obj_class_0)
    with open(data_set_path + dir_path_class_1 + 'root.txt', "r") as fr:  #load roots from class 1
        lines = fr.readlines()
        for line in lines:
            tirp_obj_class_1 = TIRP()
            tirp_obj_class_1.__dict__.clear()
            tirp_obj_class_1.__dict__.update(json.loads(line))
            root_elements_class_1.append(tirp_obj_class_1)
    for root_element_class_1 in root_elements_class_1:
        file_name = states_by_name[root_element_class_1.get_symbols()[0]] + '.txt'
        if not root_element_class_1.get_exist_in_class_0():  # if the root not exist in class 0
            with open(data_set_path + dir_path_class_0 + 'root.txt', "a") as fr:
                r = json.dumps(root_element_class_1, default=lambda x: x.__dict__)
                root_element_class_1.set_class_0_properties(min_ver_support)
                fr.write("%s\n" % r)
                temp_list.append(root_element_class_1)  # insert it to temp list
            if os.path.isfile(data_set_path + dir_path_class_1 + file_name):  # if the root has childs
                with open(data_set_path + dir_path_class_1 + file_name, "r") as fr:
                    tirp = fr.readlines()
                    with open(data_set_path + dir_path_class_0 + file_name, "a") as fs:
                        fs.writelines(tirp)  # insert all the branch to the tree
        else:  # if the root exist in class 0
            for root_element_class_0 in root_elements_class_0:
                if root_element_class_0.get_symbols()[0] == root_element_class_1.get_symbols()[0]:  #find the right root in class 0
                    root_element_class_0.set_exist_in_class_0() # update root elements that were found to true
                    if len(root_element_class_0.get_childes()) == 0:
                        if os.path.isfile(data_set_path + dir_path_class_1 + file_name):
                            root_element_class_0.set_childes(has_childs=True)
                    break
            if os.path.isfile(data_set_path + dir_path_class_1 + file_name):  # if the root has childs
                with open(data_set_path + dir_path_class_1 + file_name, "r") as fr:  #load the branch
                    tirp_class_1 = fr.readline()
                    tirp_dict_class_1 = json.loads(tirp_class_1)
                    tirp_obj_class_1 = TIRP()
                    tirp_obj_class_1.__dict__.clear()
                    tirp_obj_class_1.__dict__.update(tirp_dict_class_1)
                if os.path.isfile(data_set_path + dir_path_class_0 + file_name):  # if the branch exist in class 0
                    with open(data_set_path + dir_path_class_0 + file_name, "r") as fr:  #load the branch
                            tirp_class_0 = fr.readline()
                            tirp_dict_class_0 = json.loads(tirp_class_0)
                            tirp_obj_class_0 = TIRP()
                            tirp_obj_class_0.__dict__.clear()
                            tirp_obj_class_0.__dict__.update(tirp_dict_class_0)
                            tirp_obj_class_0.set_exist_in_class_0()
                            # if os.path.isfile(data_set_path + '/tempChunks1/' + file_name): # if the branch exist in class 1
                            tirp_obj_class_0 = insert_tirp_to_tree(tirp_obj_class_1, tirp_obj_class_0, min_ver_support)  #combine the branches
                else:  # if the branch not exist in class 0 but the root exist
                    if root_element_class_0.get_symbols()[0] == root_element_class_1.get_symbols()[0]:
                        if len(root_element_class_0.get_childes()) > 0 and type(root_element_class_0.get_childes()[0]) == bool:
                            root_element_class_0.update_childs(list())
                        tirp_obj_class_0 = insert_tirp_to_tree(tirp_obj_class_1, root_element_class_0, min_ver_support)
                    else:
                        if os.path.isfile(data_set_path + dir_path_class_1 + file_name):
                            tirp_obj_class_0 = tirp_obj_class_1
                        else:
                            tirp_obj_class_0 = root_element_class_1
                tirp_obj_class_0_json = json.dumps(tirp_obj_class_0, default=lambda x: x.__dict__)
                with open(data_set_path + dir_path_class_0 + file_name, "w") as fs:
                    fs.write(tirp_obj_class_0_json)
    os.remove(data_set_path + dir_path_class_0 + 'root.txt')
    with open(data_set_path + dir_path_class_0 + 'root.txt', "a") as fr:
        for root_element_class_0 in root_elements_class_0:
            r = json.dumps(root_element_class_0, default=lambda x: x.__dict__)
            fr.write("%s\n" % r)
        for root_element_class_1 in temp_list:
            r = json.dumps(root_element_class_1, default=lambda x: x.__dict__)
            fr.write("%s\n" % r)


def insert_tirp_to_tree(tirp_obj_class_1, tirp_obj_class_0, min_ver_support):
    """
    This is a recursive function that gets two branches and combines them
    :param tirp_obj_class_1:
    :param tirp_obj_class_0:
    :param min_ver_support:
    :return: combined branch
    """

    childs = tirp_obj_class_1.get_childes().copy()
    for child in childs:
        if type(child) != TIRP:
            child_tirp = TIRP()
            child_tirp.__dict__.clear()
            child_tirp.__dict__.update(child)
        else:
            child_tirp = child
        if not child_tirp.get_exist_in_class_0():
            curr_tirp = tirp_obj_class_0
            childs_class_0 = tirp_obj_class_0.get_childes()
            while curr_tirp.get_tirp_size()+1 < child_tirp.get_tirp_size():  #need to go down in the tree until geting to right level to insert
                childes_new = list()
                for child_class_0 in childs_class_0:
                    if type(child_class_0) != TIRP:
                        curr_tirp = TIRP()
                        curr_tirp.__dict__.clear()
                        curr_tirp.__dict__.update(child_class_0)
                        childes_new.append(curr_tirp)
                    else:
                        childes_new.append(child_class_0)
                childs_class_0 = childes_new
                for curr_tirp in childs_class_0:
                    curr_size = curr_tirp.get_tirp_size()
                    new_symbols = child_tirp.get_symbols()[:curr_size]
                    num_of_new_rels = int(curr_size * (curr_size - 1) / 2)
                    new_rels = child_tirp.get_rels()[:num_of_new_rels]
                    if curr_tirp.get_symbols() == new_symbols:
                        if curr_tirp.get_rels() == new_rels:
                            # if curr_tirp.get_tirp_size()+1 < child_tirp.get_tirp_size() and curr_tirp.get_exist_in_class_0():
                            childs_class_0 = curr_tirp.get_childes()
                            break  # the right level to insert found
            if not curr_tirp.get_exist_in_class_0():
                for i in range(0, len(childs_class_0)):
                    if type(childs_class_0[i]) != TIRP:
                        tirp = TIRP()
                        tirp.__dict__.clear()
                        tirp.__dict__.update(childs_class_0[i])
                    else:
                        tirp = childs_class_0[i]
                    if tirp.get_symbols() == child_tirp.get_symbols() and tirp.get_rels() == child_tirp.get_rels():
                        del childs_class_0[i]
                        break
            child_tirp.set_class_0_properties(min_ver_support)
            childs_class_0.append(child_tirp)
        tirp_obj_class_0 = insert_tirp_to_tree(child_tirp, tirp_obj_class_0, min_ver_support)
    return tirp_obj_class_0
