import logging
import os
import sys
import json
import csv
import PreProccesing
import time
import Index
import ParseOutputFile
import shutil
import re
import traceback
from TIRP import TIRP
from collections import namedtuple
from KLOutputToSearchIndexFile import KLOutputToSearchIndexFile
from hugobotPropertiesBounds import hugobotPropertiesBounds
from SearchInIndexFile import SearchInIndexFile


import numpy as np
from flask import Flask, request, render_template, jsonify, Response, make_response
from flask_cors import CORS


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
    # CORS(app, origins="*", allow_headers=[
    # "Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
    # supports_credentials=True, intercept_exceptions=False)


# states = dict()
# states_by_name = dict()
# searcher

# Classification process and output page
@app.route('/upload', methods=['GET', 'POST'])
def uploaded_file():
    try:
        start = time.time()
        # if len(os.listdir('UploadingDataSets/')) != 0:
        #     for filename in os.listdir('UploadingDataSets/'):
        #         file_path = os.path.join('UploadingDataSets/', filename)
        #         shutil.rmtree(file_path)
        if request.method == 'POST':
            edit_mode = False
            data_set_name = request.form['data_set_name']
            username = request.form['username']
            if data_set_name != 'undefined' and username != 'undefined':
                path = 'UploadingDataSets/' + data_set_name
                if os.path.exists(path):
                    shutil.rmtree(path, ignore_errors=False)
                # if 'old_data_set_name' not in request.form.keys():
                #     if not os.path.exists('DataSets/' + data_set_name):
                #         os.mkdir(path)
                #     else:
                #         return jsonify(
                #             {'errMsg': 'DataSet Already Exist'}), 416
                # else:   #edit mode
                #     edit_mode = True
                #     old_data_set_name = request.form['old_data_set_name']
                #     # if old_data_set_name != data_set_name:
                #         # os.rename('DataSets/' + old_data_set_name, 'DataSets/' + data_set_name)
                #     shutil.copytree('DataSets/' + old_data_set_name, path)
                #     #shutil.rmtree('DataSets/' + old_data_set_name, ignore_errors=True)
            else:
                if data_set_name == 'undefined' and username == 'undefined':
                    return jsonify(
                        {'errMsg': 'No DataSet Name and UserName Provided'}), 416
                if data_set_name == 'undefined':
                    return jsonify(
                        {'errMsg': 'No DataSet Name Provided'}), 416
                else:
                    return jsonify(
                         {'errMsg': 'No UserName Provided'}), 416
            class_name = request.form['className']
            # if class_name == 'undefined':
            #     class_name = ''
            timestamp = request.form['timestamp']
            second_class_name = request.form['secondclassName']
            # if second_class_name == 'undefined':
            #     second_class_name = ''
            comments = request.form['comments']
            # if comments == 'undefined':
            #     comments = ''
            # files
            if edit_mode:
                settings_path = "./UploadingDataSets/" + data_set_name + "/settings.json"
                with open(settings_path, 'r') as fs:
                    settings = json.load(fs)
            min_ver_support = ''
            num_of_relations = ''
            max_gap = ''
            num_of_entities = ''
            num_of_entities_class_1 = ''
            if 'secondClassOutput' in request.files.keys():
                second_class_output = request.files['secondClassOutput']
                second_class_output_file_name = second_class_output.filename
                second_class_output.save(path + "/class1Output")
                with open(path + "/class1Output") as fp:
                    params_line = fp.readline()
                    fp.close()
                params = re.split("[;=]", params_line)
                i = 0
                while i < len(params)-1:
                    if params[i] == 'num_of_entities':
                        num_of_entities_class_1 = int(params[i+1].rstrip())
                        break
                    i += 2
                if os.path.exists(path + '/chunks1'):
                    shutil.rmtree(path + '/chunks1', ignore_errors=True)
                os.mkdir(path + '/chunks1')
                PreProccesing.create_index_files(path + "/class1Output", path + '/chunks1')
            else:
                if not edit_mode:
                    second_class_output_file_name = 'File does not exist'
                else:
                    second_class_output_file_name = settings['second_class_output_file_name']
            if 'output' in request.files.keys():
                output = request.files['output']
                output_file_name = output.filename
                output.save(path + "/KLOutput")
                with open(path + "/KLOutput") as fp:
                    params_line = fp.readline()
                    fp.close()
                params = re.split("[;=]", params_line)
                i = 0
                while i < len(params)-1:
                    if params[i] == 'min_ver_support':
                        min_ver_support = float(params[i+1].rstrip())
                    elif params[i] == 'num_relations':
                        num_of_relations = int(params[i+1].rstrip())
                    elif params[i] == 'max_gap':
                        max_gap = int(params[i+1].rstrip())
                    elif params[i] == 'num_of_entities':
                        num_of_entities = int(params[i+1].rstrip())
                    i += 2
                # make chunks
                if os.path.exists(path + '/chunks'):
                    shutil.rmtree(path + '/chunks', ignore_errors=True)
                os.mkdir(path + '/chunks')
                PreProccesing.create_index_files(path + "/KLOutput", path + '/chunks')
            else:
                if not edit_mode:
                    output_file_name = 'File does not exist'
                else:
                    output_file_name = settings['output_file_name']
                    min_ver_support = settings['min_ver_support']
                    num_of_relations = settings['num_of_relations']
                    max_gap = settings['max_gap']
                    num_of_entities = settings['num_of_entities']
                    num_of_entities_class_1 = settings['num_of_entities_class_1']
            if 'states' in request.files.keys():
                states_arr = []
                states_file = request.files['states']
                states_file_name = states_file.filename
                states_file.save(path + "/states.csv")
                with open(path + "/states.csv", encoding='utf-8-sig') as csvFile:
                    csvReader = csv.DictReader(csvFile)
                    for csvRow in csvReader:
                        csvRow.update({fieldname: value.strip() for (fieldname, value) in csvRow.items()})
                        states_arr.append(csvRow)
                # write the data to a json file
                os.remove(path + "/states.csv")
                with open(path + "/states.json", "w") as jsonFile:
                    #jsonFile.write(json.dumps(states_arr, indent=4))
                    for state in states_arr:
                        state_as_json = json.dumps(state, default=lambda x: x.__dict__)
                        jsonFile.write("%s\n" % state_as_json)
                if 'output' not in request.files.keys() and output_file_name != 'File does not exist':
                    # if user changes only state file and not the output file -> make chunks
                    if os.path.exists(path + '/chunks'):
                        shutil.rmtree(path + '/chunks', ignore_errors=True)
                    os.mkdir(path + '/chunks')
                    PreProccesing.create_index_files(path + "/KLOutput", path + '/chunks')
                    if 'secondClassOutput' not in request.files.keys() and second_class_output_file_name != 'File does not exist':
                        # need to make chunks foe class 1
                        if os.path.exists(path + '/chunks1'):
                            shutil.rmtree(path + '/chunks1', ignore_errors=True)
                        os.mkdir(path + '/chunks1')
                    PreProccesing.create_index_files(path + "/class1Output", path + '/chunks1')
            else:
                if not edit_mode:
                    states_file_name = 'File does not exist'
                else:
                    states_file_name = settings['states_file_name']
            if 'entities' in request.files.keys():
                arr = []
                entities = request.files['entities']
                entities_file_name = entities.filename
                entities.save(path + "/entities.csv")
                with open(path + "/entities.csv") as csvFile:
                    csvReader = csv.DictReader(csvFile)
                    for csvRow in csvReader:
                        csvRow.update({fieldname: value.strip() for (fieldname, value) in csvRow.items()})
                        arr.append(csvRow)
                # write the data to a json file
                os.remove(path + "/entities.csv")
                with open(path + "/entities.json", "w") as jsonFile:
                    for entity in arr:
                        entity_as_json = json.dumps(entity, default=lambda x: x.__dict__)
                        jsonFile.write("%s\n" % entity_as_json)
            else:
                if not edit_mode:
                    entities_file_name = 'File does not exist'
                else:
                    entities_file_name = settings['entities_file_name']
            if 'rawData' in request.files.keys():
                raw_data = request.files['rawData']
                raw_data_file_name = raw_data.filename
                raw_data.save(path + "/rawData")
                getHugobotBounds()
            else:
                if not edit_mode:
                    raw_data_file_name = 'File does not exist'
                else:
                    raw_data_file_name = settings['raw_data_file_name']
                if 'data_set_to_copy_from' in request.form.keys():
                    data_set_to_copy_from = request.form['data_set_to_copy_from']
                    if data_set_to_copy_from != "" and data_set_to_copy_from:
                        shutil.copyfile('DataSets/' + data_set_to_copy_from + '/propDic.json', 'UploadingDataSets/' + data_set_name + '/propDic.json')
                        if 'secondClassOutput' in request.files.keys():
                            shutil.copyfile('DataSets/' + data_set_to_copy_from + '/rawDataPerClass.json', 'UploadingDataSets/' + data_set_name + '/rawDataPerClass.json')
                            shutil.copyfile('DataSets/' + data_set_to_copy_from + '/valuesPerBins.json', 'UploadingDataSets/' + data_set_name + '/valuesPerBins.json')
                        settings_path = "./DataSets/" + data_set_to_copy_from + "/settings.json"
                        with open(settings_path, 'r') as fs:
                            settings = json.load(fs)
                        raw_data_file_name = settings['raw_data_file_name']
            #     time_intervals = request.files['timeIntervals']
            #     time_intervals_file_name = time_intervals.filename
            #     time_intervals.save(path + "/timeIntervals")
            # else:
            #     if not edit_mode:
            #         time_intervals_file_name = 'File does not exist'
            #     else:
            #         time_intervals_file_name = settings['time_intervals_file_name']
            settings = {
                'data_set_name': data_set_name,
                'username': username,
                'class_name': class_name,
                'timestamp': timestamp,
                'second_class_name': second_class_name,
                'comments': comments,
                'output_file_name': output_file_name,
                'min_ver_support': min_ver_support,
                'num_of_relations': num_of_relations,
                'max_gap': max_gap,
                'num_of_entities': num_of_entities,
                'num_of_entities_class_1': num_of_entities_class_1,
                'states_file_name': states_file_name,
                'entities_file_name': entities_file_name,
                'raw_data_file_name': raw_data_file_name,
                # 'time_intervals_file_name': time_intervals_file_name,
                'second_class_output_file_name': second_class_output_file_name,
            }
            # to save to root elements
            # global states, states_by_name
            if 'output' in request.files.keys() or 'states' in request.files.keys() or \
                    'secondClassOutput' in request.files.keys() or 'entities' in request.files.keys():
                if output_file_name != 'File does not exist':
                    if os.path.exists(path + '/tempChunks'):
                        shutil.rmtree(path + '/tempChunks', ignore_errors=True)
                    if os.path.exists(path + '/tempChunks_with_entities'):
                        shutil.rmtree(path + '/tempChunks_with_entities', ignore_errors=True)
                    os.mkdir(path + '/tempChunks')
                    os.mkdir(path + '/tempChunks_with_entities')
                    if os.path.exists(path + '/states.json'):
                        states, states_by_name = ParseOutputFile.parse_states_file(path + '/states.json')
                    if second_class_output_file_name != 'File does not exist':
                        if os.path.exists(path + '/tempChunks1'):
                            shutil.rmtree(path + '/tempChunks1', ignore_errors=True)
                        if os.path.exists(path + '/tempChunks1_with_entities'):
                            shutil.rmtree(path + '/tempChunks1_with_entities', ignore_errors=True)
                        os.mkdir(path + '/tempChunks1')
                        os.mkdir(path + '/tempChunks1_with_entities')
                        root_elements_class_1 = Index.parse_main_index(path, '/tempChunks_with_entities/', '/tempChunks1_with_entities/', states, states_by_name, 'class_1', min_ver_support, second_class_output_file_name, True)
                        with open(path + '/tempChunks1_with_entities/root.txt', "w") as fs:
                            for r in root_elements_class_1:
                                fs.write("%s\n" % r)
                        root_elements_class1_without_entities = Index.parse_main_index(path, '/tempChunks/', '/tempChunks1/', states, states_by_name, 'class_1', min_ver_support, second_class_output_file_name, False)
                        with open(path + '/tempChunks1/root.txt', "w") as fs_without:
                            for r in root_elements_class1_without_entities:
                                fs_without.write("%s\n" % r)
                    root_elements = Index.parse_main_index(path, '/tempChunks_with_entities/', '/tempChunks1_with_entities/', states, states_by_name, 'class_0', min_ver_support, second_class_output_file_name, True)
                    with open(path + '/tempChunks_with_entities/root.txt', "w") as fs:
                        for r in root_elements:
                            fs.write("%s\n" % r)
                    root_elements_without_entities = Index.parse_main_index(path, '/tempChunks/', '/tempChunks1/', states, states_by_name, 'class_0',min_ver_support,second_class_output_file_name, False)
                    with open(path + '/tempChunks/root.txt', "w") as fs_without:
                        for r in root_elements_without_entities:
                            fs_without.write("%s\n" % r)
                    if second_class_output_file_name != 'File does not exist':
                        Index.marge_trees(path, '/tempChunks/', '/tempChunks1/', states_by_name, min_ver_support)
                        Index.marge_trees(path, '/tempChunks_with_entities/', '/tempChunks1_with_entities/', states_by_name, min_ver_support)
                if os.path.exists(path + '/chunks'):
                    shutil.rmtree(path + '/chunks', ignore_errors=True)
                if os.path.exists(path + '/chunks1'):
                    shutil.rmtree(path + '/chunks1', ignore_errors=True)
                if os.path.exists(path + '/tempChunks1'):
                    shutil.rmtree(path + '/tempChunks1', ignore_errors=True)
                if os.path.exists(path + '/tempChunks1_with_entities'):
                    shutil.rmtree(path + '/tempChunks1_with_entities', ignore_errors=True)
            # create index file for search
            if output_file_name != 'File does not exist' and os.path.exists(path + '/states.json'):
                sercher = KLOutputToSearchIndexFile(input_file_path=path + "/KLOutput", output_file_path=path + "/searchIndex",
                                      num_of_entities=num_of_entities, data_set_path=path, states=states, ids_counter=1)
                if second_class_output_file_name != 'File does not exist':
                    ids_counter = sercher.get_ids_counter()
                    KLOutputToSearchIndexFile(input_file_path=path + "/class1Output", output_file_path=path + "/searchIndexClass1",
                                      num_of_entities=num_of_entities_class_1, data_set_path=path, states=states, ids_counter=ids_counter)
            # searcher = SearchInIndexFile(path + "/searchIndex")
            with open(path + '/settings.json', 'w') as outfile:
                json.dump(settings, outfile)
            # if edit_mode:
            #     shutil.rmtree('DataSets/' + old_data_set_name, ignore_errors=True)
            shutil.copytree(path, 'DataSets/' + data_set_name)
            shutil.rmtree(path, ignore_errors=True)
            data = {'status': 'ok'}
            js = json.dumps(data)
            resp = Response(js, status=200, mimetype='application/json')
            resp.headers['Access-Control-Allow-Origin'] = '*'
            # resp = jsonify(data)
            # resp.
            # resp.status_code = 200
            end = time.time()
            print(end - start)
            return resp
    except Exception as e:
        print(e)
        traceback.print_exc()
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=False)
            raise Exception('Something went wrong.')


@app.route('/getDataSets', methods=['GET'])
def get_dataSets():
    data_sets_names = os.listdir("./DataSets")
    data_sets_details = list()
    for name in data_sets_names:
        settings_path = "./DataSets/" + name + "/settings.json"
        with open(settings_path, 'r') as fs:
            settings = json.load(fs)
        data_sets_details.append(settings)
    return jsonify(
        {'DataSets': data_sets_details})


@app.route('/getDataSets', methods=['POST'])
def get_dataSets_url():
    data = request.get_json()
    data_set_name = data.get('data_set_name', '')
    # data_sets_names = os.listdir("./DataSets")
    data_sets_details = list()
    # for name in data_sets_names:
    settings_path = "./DataSets/" + data_set_name + "/settings.json"
    with open(settings_path, 'r') as fs:
        settings = json.load(fs)
    data_sets_details.append(settings)
    return jsonify(
        {'DataSets': data_sets_details})

@app.route('/getUploadCompleted', methods=['POST'])
def get_upload_completed():
    data = request.get_json()
    data_set_name = data.get('data_set_name', '')
    path = "./DataSets/" + data_set_name
    ans = False
    if os.path.exists(path):
        ans = True
    return jsonify(
        {'Exist': ans})


@app.route('/deleteDataset', methods=['POST'])
def delete_dataset():
    data = request.get_json()
    data_set_name = data.get('data_set_name', '')
    path = "./DataSets/" + data_set_name
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=False)
    return jsonify(
        {'Status': 'OK'})


@app.route('/getEntities', methods=['POST'])
def get_entities():
    if request.method == 'POST':
        data_set_name = request.form['data_set_name']
        path = 'DataSets/' + data_set_name + '/entities.json'
        with open(path, 'r') as fs:
            entities = fs.readlines()
    return jsonify({'Entities': entities})


@app.route('/getStates', methods=['POST'])
def get_states():
    # global states, states_by_name
    if request.method == 'POST':
        data_set_name = request.form['data_set_name']
        path = 'DataSets/' + data_set_name + '/states.json'
        with open(path, 'r') as fs:
            # states_from_file = json.load(fs)
            states_from_file = fs.readlines()
            # if len(states) == 0 or len(states_by_name) == 0:
            # states, states_by_name = ParseOutputFile.parse_states_file(path)
    return jsonify({'States': states_from_file})


@app.route('/initiateTirps', methods=['POST'])
def initiate_tirps():
    data_set_name = request.form['data_set_name']
    path = 'DataSets/' + data_set_name
    # global states, states_by_name
    # if os.path.exists(path + '/states.json'):
    #     states, states_by_name = ParseOutputFile.parse_states_file(path + '/states.json')
    with open(path + '/tempChunks/root.txt', "r") as fr:
        lines = fr.readlines()
    # searcher = SearchInIndexFile(path + "/searchIndex")
    response = make_response(jsonify({'Root': lines}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/getSubTree', methods=['POST'])
def get_sub_tree():
    json_TIRPS = []
    data = request.get_json()
    data_set_name = data.get('data_set_name', '')
    TIRP_from_req = data.get('TIRP', '')
    # tirp_obj = TIRP()
    # tirp_obj.__dict__.clear()
    # tirp_obj.__dict__.update(TIRP_from_req)
    path = 'DataSets/' + data_set_name
    states, states_by_name = ParseOutputFile.parse_states_file(path + '/states.json')
    # tirp = Index.get_sub_tree(tirp_obj, states, states_by_name, path)
    # s = json.dumps(tirp, default=lambda x: x.__dict__)
    # for tirp in tirps:
    #     s = json.dumps(tirp, default=lambda x: x.__dict__)
    #     json_TIRPS.append(s)
    file_name = states_by_name[TIRP_from_req] + '.txt'
    if os.path.isfile(path + '/tempChunks/' + file_name):
        with open(path + '/tempChunks/' + file_name, "r") as fr:
            tirp = fr.readlines()
    # else:
    #     tirp = json.dumps(TIRP_from_req)
    response = make_response(jsonify({'TIRPs': tirp}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/getHugobotBounds', methods=['POST'])
def getHugobotBounds():
    data_set_name = request.form['data_set_name']
    if os.path.isdir('UploadingDataSets/' + data_set_name):
        prop_dic_path = 'UploadingDataSets/' + data_set_name + '/propDic.json'
    else:
        prop_dic_path = 'DataSets/' + data_set_name + '/propDic.json'

    if os.path.isfile(prop_dic_path):
        with open(prop_dic_path, "r") as out_file:
            return jsonify(out_file.readlines())

    raw_data_path = 'UploadingDataSets/' + data_set_name + '/rawData'
    states_file_path = 'UploadingDataSets/' + data_set_name + '/states.json'

    with open(states_file_path, 'r') as fs:
        states_list = []
        states_from_file = fs.readlines()
        for line in states_from_file:
            states_list.append(json.loads(line))

    hugobotPropertiesBounds(raw_data_path, states_list, 'UploadingDataSets/' + data_set_name,\
                                                    hugobot_path="C:/Users/GUY/Desktop/hugobotVersions/new_kfir/cli.py")


@app.route('/getValuesPerBinsDic', methods=['POST'])
def get_values_per_entity():
    data_set_name = request.form['data_set_name']
    entity_values_dic_path = 'DataSets/' + data_set_name + '/valuesPerBins.json'

    if os.path.isfile(entity_values_dic_path):
        with open(entity_values_dic_path, "r") as out_file:
            return jsonify(out_file.readlines())

    return "the file doesn't exists"


@app.route('/getValuesPerClass', methods=['POST'])
def get_values_per_class():
    data_set_name = request.form['data_set_name']
    entity_values_dic_path = 'DataSets/' + data_set_name + '/rawDataPerClass.json'

    if os.path.isfile(entity_values_dic_path):
        with open(entity_values_dic_path, "r") as out_file:
            return jsonify(out_file.readlines())

    return "the file doesn't exists"


@app.route('/searchTirps', methods=['POST'])
def searchTirps():
    data = request.get_json()
    data_set_name = data.get('data_set_name', '')
    path = 'DataSets/' + data_set_name
    search_in_class_1 = data.get('search_in_class_1', '')
    startsList = data.get('startsList', '')
    containList = data.get('containList', '')
    endsList = data.get('endsList', '')
    minHS = data.get('minHS', '')
    maxHS = data.get('maxHS', '')
    minVS = data.get('minVS', '')
    maxVS = data.get('maxVS', '')
    searcher = SearchInIndexFile(path + "/searchIndex")
    results = searcher.get_serached_tirps(start_sym=startsList, contains_sym=containList, end_sym=endsList, min_m_hs=minHS,
                                max_m_hs=maxHS, min_vs=minVS, max_vs=maxVS)
    if search_in_class_1:
        searcher_class_1 = SearchInIndexFile(path + "/searchIndexClass1")
        results_class_1 = searcher_class_1.get_serached_tirps(start_sym=startsList, contains_sym=containList, end_sym=endsList,
                                              min_m_hs=minHS,
                                              max_m_hs=maxHS, min_vs=minVS, max_vs=maxVS)
        results_set = set()
        for result in results:
            result_arr = result.split(',')
            results_set.add(result_arr[0]+result_arr[1])
        for result in results_class_1:
            result_arr = result.split(',')
            result_str = result_arr[0]+result_arr[1]
            if result_str not in results_set:
                results.append(result)
    response = make_response(jsonify({'Results': results}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/find_Path_of_tirps', methods=['POST'])
def find_Path_of_tirps():
    # global states, states_by_name
    json_path_of_tirps = []
    data = request.get_json()
    data_set_name = data.get('data_set_name', '')
    data_Set_Path = 'DataSets/' + data_set_name
    # if len(states) == 0 or len(states_by_name) == 0:
    states, states_by_name = ParseOutputFile.parse_states_file(data_Set_Path + '/states.json')
    symbols = data.get('symbols', '')
    relations = data.get('relations', '')
    to_add_entities = data.get('to_add_entities', '')
    path_of_tirps = Index.find_Path_of_tirps(symbols=symbols, rels=relations, data_set_path=data_Set_Path, states=states,states_by_name=states_by_name, to_add_entities=to_add_entities)
    for tirp in path_of_tirps:
        json_path_of_tirps.append(json.dumps(tirp, default=lambda x: x.__dict__))
    response = make_response(jsonify({'Path': json_path_of_tirps}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == '__main__':
    handler = logging.FileHandler('./log.log')
    handler.setLevel(logging.ERROR)
    app.debug = True  # allows for changes to be enacted without rerunning server
    app.logger.addHandler(handler)
    app.config["JSON_SORT_KEYS"] = False
    app.run()
