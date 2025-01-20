import os 
from os import path 
import numpy as np

file_list = []
def scaner_file (url):
    file  = os.listdir(url)
    for f in file:
        real_url = path.join (url , f)
        if path.isfile(real_url):
            file_list.append(real_url)
            # return
            # print(real_url)
        elif path.isdir(real_url):
            scaner_file(real_url)
        else:
            print("其他情况")
            pass

# scaner_file("./new_log")

def read_results(file):
    with open(file, "r") as f:
        data = f.read()

    i = -1
    j = data.rfind('around ')
    acc_list = []
    while i < j:
        start = i+1
        i = data.index('around ', start)
        acc = float(data[i+7:i+14])
        acc_list.append(acc)

    acc_list = np.array(acc_list)

    print(acc_list)

read_results('')