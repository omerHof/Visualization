import fileinput
import sys

input_path = 'C:/Users/GUY/Desktop/dataSets/AHE/hugo_3000/KL-class-1.0.txt'
temp_path = 'C:/Users/GUY/Desktop/dataSets/AHE/control_cohort_KB/class1_600_abstructed_with_offset.txt'


def pares(input_path):
    index=0
    file = open(input_path, "r+")
    t = open(temp_path, "w")
    line = file.readline()
    t.write(line)
    t.write(file.readline())
    line = file.readline()
    while line:
        line = line[0:line.index(';')+1]+ '\n'
        # +','+str(index)+';0;10000\n'
        t.write(line)
        t.write(file.readline())
        line = file.readline()
        index= index + 1
    file.close()


def parse_kl_input(input_path, output_path):
    index = 0
    file = open(input_path, "r+")
    t = open(output_path, "w")
    t.write(file.readline())  # startToncepts
    t.write(file.readline())  #numberOfEntities
    entity_id_line = file.readline()
    while entity_id_line:
        entity_details_line = file.readline()
        min, max = find_max_and_min(entity_details_line)
        entity_id_line = entity_id_line[0:entity_id_line.index(';')] +','+str(index)+';'+min+';'+max + '\n'
        t.write(entity_id_line)
        t.write(entity_details_line)
        entity_id_line = file.readline()
        index += 1
    file.close()


def find_max_and_min(line):
    min = sys.maxsize
    max = -sys.maxsize
    line_arr = line.split(";")
    for sec in line_arr:
        sec_arr = sec.split(",")
        if len(sec_arr) != 4:
            continue
        start_time = int(sec_arr[0])
        if start_time < min:
            min = start_time
        end_time = int(sec_arr[1])
        if end_time > max:
            max = end_time
    return str(0), str(max)


parse_kl_input(input_path, temp_path)
