import argparse
import math
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="Input CSV File Path")
parser.add_argument("--k", help="Input the k  for the kNN classifier")
args = parser.parse_args()




def classification(list_dist,list_obj):
    # Finding the class label as per kNN
    DKNN = list_dist[len(list_dist)-1]
    D1NN = list_dist[0]
    weight_A = 0
    weight_B = 0
    for l in range(len(list_dist)):
        if list_obj[l][0] == u[0]:
            weight_A += weight_calcu(DKNN,D1NN,list_dist[l])
        else:
            weight_B += weight_calcu(DKNN,D1NN,list_dist[l])
    if weight_A > weight_B:
        return u[0]
    else:
        return u[1]

def weight_calcu(DKNN,D1NN,DiNN):
    # Calculating the weight as per weighting scheme
    if DKNN == D1NN:
        return 1
    else:
        weight = (DKNN-DiNN)/(DKNN-D1NN)
        return weight

def kNN_Classifier(input_data,k,classify = False):
    #insert first case base auto acc to IB2
    case_base = [input_data[0]]
    list_obj_others = list()
    misclassification = 0
    # This loop is for forming the case_base
    for x in range(1,len(input_data)):
        dist = list()
        dictionary = {}
        list_obj = list()
        for u in range(len(case_base)):
            d = eucledian_distance(input_data[x],case_base[u])
            dist.append(d)
            dictionary[d] = case_base[u]
        dist = sorted(dist)
        size = min(len(case_base),1)
        for z in range(size):
            list_obj.append(dictionary[dist[z]])
        label = classification(dist[0:size], list_obj)
        if label != input_data[x][0]:
            case_base.append(input_data[x])
        else:
            list_obj_others.append(input_data[x])

    for f in range(len(list_obj_others)):
        # This loop is for finding number of misclassification with respective to above case_base
        dist1 = list()
        dictionary1 = {}
        list_obj1 = list()
        for g in range(len(case_base)):
            d1 = eucledian_distance(list_obj_others[f],case_base[g])
            dist1.append(d1)
            dictionary1[d1] = case_base[g]
        dist1 = sorted(dist1)
        for h in range(k):
            list_obj1.append(dictionary1[dist1[h]])
        label1 = classification(dist1[0:k], list_obj1)
        if label1 != list_obj_others[f][0]:
            misclassification += 1
    if classify:
        return misclassification
    else:
        return case_base


def eucledian_distance(case_A,case_B):
    # Calculating the eucledian_distance between two objects as per eucledian metric
    squared_diff = 0
    for a in range(1,len(case_A)):
        squared_diff += (case_A[a]-case_B[a])**2
    dist = math.sqrt(squared_diff)
    return dist

csv_file_path = args.data
value_of_k = int(args.k)

# input csv
input_data = list()
with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=",")
    data_array = np.array(list(reader))
    for row in data_array:
        x = [float(x) for x in row[1:3]]
        data = [row[0]]
        data.extend(x)
        input_data.append(data)

c = data_array[:, 0]

u, counts = np.unique(c, return_counts= True)

output = kNN_Classifier(input_data,1,False)

misclass  = kNN_Classifier(input_data,value_of_k,True)

print(misclass)

for i in range(len(output)):
    print(",".join(repr(e) for e in output[i]).replace("'", ''))
