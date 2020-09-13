import csv
import sys
import pandas as pd
from Entity import Entity
from collections import Counter, defaultdict
import json


entities = dict()
values_per_property = dict()

def parse_entities_file():
    """
    This function parse the entities file and creates set of entities objects. The file must contain a column named id
    :return:
    """
    file_name = './entities.csv'
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        if 'id' not in csv_reader.fieldnames:
            sys.exit("Error! Start time can't be equal to end time! please change KLC code")
        for row in csv_reader:
            properties = dict()
            for prop in csv_reader.fieldnames:
                if prop == 'id':
                    entityId = row[prop].rstrip()
                else:
                    properties[prop] = row[prop].rstrip()
            entity = Entity(entityId=entityId, properties=properties)
            entities[entityId] = entity


def get_values_per_property():
    file_name = './entities.csv'
    with open(file_name, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        gapminder = pd.read_csv(file_name)
        for prop in csv_reader.fieldnames:
            values_per_property[prop] = gapminder[prop].unique()



def parse2(path, ids):
    res = dict()
    try:
        with open(path + 'entities.json') as f:
            j = json.load(f)
            c = Counter()
            ans = dict()
            for d in j:
                for k, v in d.items():
                    key = k.rstrip()
                    val = v.rstrip()
                    if key == 'id':
                        continue
                    if d['id'] in ids:
                        c.update([key + '-' + val])
                    l = res.get(key)
                    if l:
                        if val not in l:
                            res[key].append(val)
                    else:
                        res[key] = list()
                        res[key].append(val)
            for k in res.keys():
                v = res[k].copy()
                res[k].clear()
                for prop in v:
                    prop_dict = dict()
                    prop_name = k + '-' + prop
                    if prop_name in c.keys():
                        prop_dict[prop] = c[prop_name]
                    else:
                        prop_dict[prop] = 0
                    res[k].append(prop_dict)
            return res
    except:
        return res





ids = list()
ids.append('1')
ids.append('2')
ids.append('3')
parse2('DataSets/out/', ids)
parse_entities_file()
get_values_per_property()
