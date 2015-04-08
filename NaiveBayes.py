import sys
import time
import random
import math

#print 'Number of arguments:', len(sys.argv), 'arguments.'
#print 'Argument List:', str(sys.argv)

train_file = sys.argv[1]
test_file = sys.argv[2]
# print("train_file = " + train_file)
# print("test_file = " + test_file)

#test_file = "dataset/led.test"
#train_file = "dataset/led.train"



start_time = time.time()

no_of_attr = []
attr_list = []
max_attr_train = 1
max_attr_test = 1
c_pos1 = {}
c_neg1 = {}
PofC_train = {}
attr_dict_train = {}
trainfile_line_ctr = 0
trainfile_class_list = []
for line in open(train_file):
    if len(line) > 5:
        attr_list = []
        trainfile_line_ctr += 1
        stuff = line.split()
        c_value = int(stuff[0])
        trainfile_class_list.append(c_value)
        if PofC_train.has_key(int(stuff[0])):
            PofC_train[c_value] += 1
        else:
            PofC_train[c_value] = 1
        for i in range(1, len(stuff)):
            key = stuff[i]
            if c_value == 1:
                if c_pos1.has_key(key):
                    c_pos1[key] += 1
                else:
                    c_pos1[key] = 1
            else:
                if c_neg1.has_key(key):
                    c_neg1[key] += 1
                else:
                    c_neg1[key] = 1

            temp = stuff[i].split(":")[0]
            attr_list.append(int(temp))
            max_attr_list = max(attr_list)
            if (max_attr_list > max_attr_train):
                max_attr_train = max_attr_list

C_pos_count = PofC_train[1]
# print("Pos count = " + str(C_pos_count))
C_neg_count = PofC_train[-1]
# print("Neg count = " + str(C_neg_count))


# for TEST file
TP_list_Test = []
FP_list_Test = []
TN_list_Test = []
FN_list_Test = []

PofC_test = {}
a3 = b3 = 1
a4 = b4 = 1
testfile_line_ctr = 0
attr_dict_test = {}
testfile_class_list = []
for line in open(test_file):
    attr_list_test = []
    testfile_line_ctr += 1
    stuff = line.split()
    testfile_class_list.append(int(stuff[0]))
    for i in range(1, len(stuff)):
        temp = stuff[i].split(":")[0]
        attr_list_test.append(int(temp))
        attr = stuff[i]
        if attr_dict_test.has_key(attr):
            attr_dict_test[attr] += 1
        else:
            attr_dict_test[attr] = 1
        max_attr_list = max(attr_list_test)
        if (max_attr_list > max_attr_test):
            max_attr_test = max_attr_list

max_attr = max(max_attr_test, max_attr_train)
a2 = b2 = 1

line = ""
for line in open(train_file):
    attr_list_train = []
    stuff = line.split()
    if(len(stuff) > 0):
        c_value = int(stuff[0])
        for i in range(1, len(stuff)):
            temp = stuff[i].split(":")[0]
            attr_list_train.append(int(temp))
        for i in range(1, max_attr+1):
            if i not in attr_list_train:
                key_str = str(i) + ":0"

                if c_value == 1:
                    if c_pos1.has_key(key_str):
                        c_pos1[key_str] += 1
                    else:
                        c_pos1[key_str] = 1
                else:
                    if c_neg1.has_key(key_str):
                        c_neg1[key_str] += 1
                    else:
                        c_neg1[key_str] = 1


prob_of_attr_train = []
prob_of_C_pos = float(PofC_train[1]) / trainfile_line_ctr
prob_of_C_neg = float(PofC_train[-1]) / trainfile_line_ctr

a1 = b1 = 1
accuracy_list_Train = []
accuracy_list_Test = []
TP_list_Train = []
FP_list_Train = []
TN_list_Train = []
FN_list_Train = []

#calculate Naive Bayes
def calc_naive_bayes(test_file, testfile_class_list):
    predicted_class_list = []
    #print("Naive Bayes -------------------" + str(test_file))
    for line in open(test_file):
        if len(line) > 5 :
            attr_list = []
            attr_list_2 = []
            c_pos_prob_list = []
            c_neg_prob_list = []
            stuff = line.split()
            for i in range(1, len(stuff)):
                temp = stuff[i].split(":")[0]
                attr_list.append(int(temp))
                attr_list_2.append(stuff[i])
            # print(attr_list)
            for i in range(1, max_attr+1):
                if i not in attr_list:
                    key_str = str(i) + ":0"
                    #print("key_str = " + str(key_str))
                    attr_list_2.append(key_str)
            # print("final attr list = ", attr_list_2)
            final_cond_prob_pos = 1
            final_cond_prob_neg = 1
            global a2, a4
            global b2, b4

            for key in attr_list_2:
                if c_pos1.has_key(key):
                    cond_prob_pos = float(c_pos1[key])/C_pos_count
                    final_cond_prob_pos*=cond_prob_pos
                    c_pos_prob_list.append(cond_prob_pos)

                if c_neg1.has_key(key):
                    cond_prob_neg = float(c_neg1[key])/C_neg_count
                    final_cond_prob_neg*=cond_prob_neg
                    c_neg_prob_list.append(cond_prob_neg)


            final_cond_prob_pos*=prob_of_C_pos
            final_cond_prob_neg*=prob_of_C_neg
            predicted_class = 0
            if final_cond_prob_pos > final_cond_prob_neg :
                predicted_class = 1
            else :
                predicted_class = -1

            predicted_class_list.append(predicted_class)

    nb_true_pos = 0
    nb_true_neg = 0
    nb_false_pos = 0
    nb_false_neg = 0
    
    for i in range(0, len(predicted_class_list)):
        if predicted_class_list[i]==testfile_class_list[i] :
            if predicted_class_list[i] == 1:
                nb_true_pos+=1
            else :
                nb_true_neg+=1

        else :
            if predicted_class_list[i] > testfile_class_list[i]:
                nb_false_pos+=1

            else :
                nb_false_neg+=1


    print(str(nb_true_pos) + " " + str(nb_false_neg) + " " + str(nb_false_pos) + " " + str(nb_true_neg) )

    return  predicted_class_list

train_predicted_class_list = calc_naive_bayes(train_file, trainfile_class_list)

predicted_class_list = calc_naive_bayes(test_file, testfile_class_list)

