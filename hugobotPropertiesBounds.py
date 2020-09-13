import os
import shutil
import itertools
import pandas as pd
import json
from Property import Property


class hugobotPropertiesBounds:
    """ This class gets a path to the rawData file, to the states file and to the hugobot
        then it uses the hugoBot to create a file with the discretization of the proprties(every possible discretization)
        and gets from it all the bounds of the discretization.
        the discretization bounds are saved in a dictionary"""

    _ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
    # _HUGOBOT_OUTPUT_DIR_PATH = "HugoOutEW"
    # _PREPROCESSING_FILE_PATH = "aid_files/preprocessing.csv"
    # _TEMPORAL_ABSTRACTIOM_FILE_PATH = "aid_files/temporal_abstraction.csv"

    _PROPERTY_ID_INDEX = 1
    _TIMESTAMP_INDEX = 2
    _PROPERTY_VALUE_INDEX = 3
    _ENTITY_ID_INDEX = 0

    _START_HUGOBOT_COMMAND = ""

    ''' 'equal-width', 'equal-frequency', 'sax', 'persist', 'td4c-cosine' , 
    'td4c-entropy', 'td4c-entropy-ig', 'td4c-skl', 'td4c-diffsum', 'td4c-diffmax']'''

    _discretization_order_list = None
    _discretization_bounds_dic = None

    def __init__(self, raw_data_path, states_json, output_dir_path, hugobot_path):
        os.makedirs(output_dir_path + "/aid_files", mode=0o777, exist_ok=True)  #####
        self._HUGOBOT_OUTPUT_DIR_PATH = output_dir_path + "/HugoOutEW"  ##########
        self._PREPROCESSING_FILE_PATH = output_dir_path + "/aid_files/preprocessing.csv"  ###########3
        self._TEMPORAL_ABSTRACTIOM_FILE_PATH = output_dir_path + "/aid_files/temporal_abstraction.csv"  ######
        self._raw_data_path = raw_data_path
        self._states_list = states_json
        self._hugobot_path = hugobot_path
        self._output_dir_path = output_dir_path
        self._properties_dic = {}
        self._discretization_bounds_dic = {}
        self._discretization_order_list = []
        self._discretization_type_list = ['equal-width', 'equal-frequency', 'sax', 'persist']

        self._entity_class = {}
        self._data_per_class = {}
        self._values_per_bin = {}

        self.raw_date_to_properties()
        self.save_values()

        self.create_files_for_hugobot()

        self.create_query_for_hugobot()
        self.execute_hugobot()

        self.set_bounds_from_file()
        self.count_values_per_bin()
        self.save_dictionaries_to_dir()
        self.delete_files()

    def raw_date_to_properties(self):
        """ This method creates all the relevant properties from the states list """
        ids_list = []
        for prop in self._states_list:
            id = prop['TemporalPropertyID']
            ids_list.append(id)
            self.add_to_properties_dic(id)
        self.set_bins_num(ids_list)

    def set_bins_num(self, ids_lst):
        """This method sets the number of bins to the properties objects"""
        for id in self._properties_dic.keys():
            self._properties_dic[id].setBinNum(ids_lst.count(id))

    def save_values(self):
        """ This method saves all the properties values to the objects """
        file = open(self._raw_data_path, 'r')
        file.readline()
        line = file.readline()[:-1].split(',')
        entities_id = {}

        while line is not None and line != ['']:

            if line[self._PROPERTY_ID_INDEX] == '-1':
                self._entity_class[line[self._ENTITY_ID_INDEX]] = float(line[self._PROPERTY_VALUE_INDEX])
            else:
                self._properties_dic[line[self._PROPERTY_ID_INDEX]].addValue(float(line[self._PROPERTY_VALUE_INDEX]))
                if line[self._ENTITY_ID_INDEX] not in entities_id.keys():
                    entities_id[line[self._ENTITY_ID_INDEX]] = True
            line = file.readline()[:-1].split(',')
        # checks if is there class information about all entities
        if len(self._entity_class) == len(entities_id):
            self.save_class_data_per_prop()
            self._discretization_type_list += ['td4c-cosine', 'td4c-entropy',
                                               'td4c-entropy-ig', 'td4c-skl', 'td4c-diffsum', 'td4c-diffmax']

    # def set_entities_classes(self, file, line):
    #     "This method creates a dictionary that contains the classes of each entity"
    #     while line is not None and line != ['']:
    #         if line[1] == '-1':
    #             self._entity_class[line[0]] = line[3] ###############3
    #             line = file.readline()[:-1].split(',')
    #
    #     self.save_class_data_per_prop()

    def save_class_data_per_prop(self):
        "This method reads from the raw data the values of each property class pair"
        try:
            file = open(self._raw_data_path, 'r')
            file.readline()
            line = file.readline()[:-1].split(',')

            while line != None and line != ['']:
                if line[1] == '-1':
                    line = file.readline()[:-1].split(',')
                    continue
                key = (line[1], self._entity_class[line[0]])
                if key not in self._data_per_class.keys():
                    self._data_per_class[key] = [float(line[3])]
                else:
                    self._data_per_class[key] += [float(line[3])]
                line = file.readline()[:-1].split(',')
        finally:
            file.close()

    def add_to_properties_dic(self, id):
        """ This method enters a certain property to the object dictionary """
        if id not in self._properties_dic.keys():
            self._properties_dic[id] = Property(id)

    def create_files_for_hugobot(self):
        """ This method creates the nescery files for the property discretization for the hugoBot"""
        self.create_pre_processing_file()
        self.create_temporal_abstraction_file()

    def create_pre_processing_file(self):
        """ This method creates the preprocessing file for the hugoBot """
        try:
            path = self._ROOT_DIR + '/' + self._PREPROCESSING_FILE_PATH
            open(path, "w")
            file = open(path, "a")
            file.write("TemporalPropertyID,PAAWindowSize,StdCoefficient,MaxGap\n")

            for prop_id in self._properties_dic.keys():
                line = str(prop_id) + ',,,1\n'
                file.write(line)
        finally:
            file.close()

    def create_temporal_abstraction_file(self):
        """ This method creates the temporal abstraction file for the hugoBot """
        try:
            path = self._ROOT_DIR + '/' + self._TEMPORAL_ABSTRACTIOM_FILE_PATH
            open(path, "w")
            file = open(path, "a")
            file.write("TemporalPropertyID,Method,NbBins,GradientWindowSize\n")

            for prop_id in self._properties_dic.keys():
                for method in self._discretization_type_list:
                    self._discretization_order_list += [(prop_id, method)]
                    line = str(prop_id) + ',' + method + ',' + str(self._properties_dic[prop_id].getBinNum()) + ',\n'
                    file.write(line)
        finally:
            file.close()

    def create_query_for_hugobot(self):
        """ This method creates the query for the hugobot to run all the discretization """

        # self._START_HUGOBOT_COMMAND += '/home/shitritg/anaconda3/bin/python "'
        self._START_HUGOBOT_COMMAND += 'python "'
        self._START_HUGOBOT_COMMAND += self._hugobot_path + '" '
        self._START_HUGOBOT_COMMAND += 'temporal-abstraction "'
        self._START_HUGOBOT_COMMAND += self._ROOT_DIR + '/' + self._raw_data_path + '" "'
        self._START_HUGOBOT_COMMAND += self._ROOT_DIR + '/' + self._HUGOBOT_OUTPUT_DIR_PATH + '" '
        self._START_HUGOBOT_COMMAND += 'per-property "'
        self._START_HUGOBOT_COMMAND += self._ROOT_DIR + '/' + self._PREPROCESSING_FILE_PATH + '" "'
        self._START_HUGOBOT_COMMAND += self._ROOT_DIR + '/' + self._TEMPORAL_ABSTRACTIOM_FILE_PATH + '"'

    def execute_hugobot(self):
        """ This method executes the hugobot """
        # os.system("if not exist " + self._HUGOBOT_OUTPUT_DIR_PATH + " mkdir " + self._HUGOBOT_OUTPUT_DIR_PATH)
        os.makedirs(self._HUGOBOT_OUTPUT_DIR_PATH, mode=0o777, exist_ok=True)  ###############
        os.system(self._START_HUGOBOT_COMMAND)

    def set_bounds_from_file(self):
        """ This method gets the all the bounds value from the hugobot output file
            and sets them to the relevant property in the object dictionary"""

        file = pd.read_csv(self._ROOT_DIR + '/' + self._HUGOBOT_OUTPUT_DIR_PATH + '/states.csv')
        bounds_list = list(file.get("BinHigh"))
        number_of_bins = self.get_hogubot_output_bins_number(list(file.get("BinID")))

        for key_tup, bins_num in zip(self._discretization_order_list, number_of_bins):
            self._properties_dic[key_tup[0]].setBinNum(bins_num)
            prop = self._properties_dic[key_tup[0]]
            prop.addBounds(key_tup[1], [round(elem, 2) for elem in bounds_list[:bins_num - 1]])
            bounds_list = bounds_list[bins_num:]

    def get_hogubot_output_bins_number(self, bins):

        counter = 0
        number_of_bins = []

        for bin_id in bins:
            if bin_id == 0:
                number_of_bins.append(counter)
                counter = 1
            else:
                counter += 1
        number_of_bins.append(counter)
        return number_of_bins[1:]

    def delete_files(self):
        """ This method deletes all the unnecessary files """
        os.remove(self._PREPROCESSING_FILE_PATH)
        os.remove(self._TEMPORAL_ABSTRACTIOM_FILE_PATH)
        shutil.rmtree(self._ROOT_DIR + '/' + self._HUGOBOT_OUTPUT_DIR_PATH)
        shutil.rmtree(self._output_dir_path + "/aid_files")  ########

    # def count_values_per_bin(self):
    #     """This method counts per each bin how many values it contains"""
    #     if self._data_per_class is {}:
    #         return
    #
    #     for key in self._data_per_class.keys():
    #         list.sort(self._data_per_class[key])
    #         values_list = self._data_per_class[key]
    #         for dis in self._discretization_type_list:
    #             new_key = str((key[0], dis, key[1]))
    #             self._values_per_bin[new_key] = [0]*self._properties_dic[key[0]].getBinNum()
    #             bounds_list = self._properties_dic[key[0]].getBoundsDic()[dis]
    #             bounds_list.append(float('inf'))
    #             prev_bound = float('-inf')
    #             cur_bound_index = 0
    #             value_counter = 0
    #             # This loop counts the number of values between two adjusted bounds
    #             for val in values_list:
    #                 if prev_bound <= val < bounds_list[cur_bound_index]:
    #                     value_counter += 1
    #                 else:
    #                     self._values_per_bin[new_key][cur_bound_index] = value_counter
    #                     prev_bound = bounds_list[cur_bound_index]
    #                     cur_bound_index += 1
    #                     value_counter = 1
    #
    #             self._values_per_bin[new_key][cur_bound_index] = value_counter
    #
    #             bounds_list.remove(float('inf'))

    def count_values_per_bin(self):
        """This method counts per each bin how many values it contains"""
        if self._data_per_class is {}:
            return

        for key in self._data_per_class.keys():
            list.sort(self._data_per_class[key])
            values_list = self._data_per_class[key]
            for dis in self._discretization_type_list:
                new_key = str((key[0], dis, key[1]))
                self._values_per_bin[new_key] = [0]*self._properties_dic[key[0]].getBinNum()
                bounds_list = self._properties_dic[key[0]].getBoundsDic()[dis]
                bounds_list.append(float('inf'))
                prev_bound = float('-inf')
                cur_bound_index = 0
                value_counter = 0
                # This loop counts the number of values between two adjusted bounds
                for val in values_list:
                    if prev_bound <= val < bounds_list[cur_bound_index]:
                        value_counter += 1
                    else:
                        self._values_per_bin[new_key][cur_bound_index] = value_counter
                        prev_bound = bounds_list[cur_bound_index]
                        cur_bound_index += 1
                        value_counter = 1

                self._values_per_bin[new_key][cur_bound_index] = value_counter
                bounds_list.remove(float('inf'))

    def save_dictionaries_to_dir(self):
        """ This method saves all the the relevant JSON files"""
        try:
            out_file = open(self._output_dir_path + '/propDic.json', 'w')
            properties_dic = json.dumps(self._properties_dic, default=lambda x: x.__dict__)
            out_file.write(properties_dic)
            out_file.close()

            if len(self._data_per_class) == 0:
                return

            out_file = open(self._output_dir_path + '/valuesPerBins.json', 'w')
            # json.dump(self._values_per_bin, out_file, default=lambda o: o.__dict__)
            properties_dic = json.dumps(self._values_per_bin, default=lambda x: x.__dict__)
            out_file.write(properties_dic)
            out_file.close()

            out_file = open(self._output_dir_path + '/rawDataPerClass.json', 'w')
            properties_dic = json.dumps(self.get_data_per_class_to_write(), default=lambda x: x.__dict__)
            out_file.write(properties_dic)
        finally:
            out_file.close()

    def get_data_per_class_to_write(self):
        """This method creates a copy of the data pre class dictionary"""
        copy = {}
        for key in self._data_per_class:
            copy[str(key)] = self._data_per_class[key]
        return copy

    def get_prop_dic(self):
        """ This function returns the dictionary that contains all the properties objects"""
        return self._properties_dic

    def get_classes_prop_dic(self):
        """This method returns the dictionary that contains the list of values per bin"""
        return self._values_per_bin
