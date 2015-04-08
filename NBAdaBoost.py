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

#test_file = "dataset/poker.test"
#train_file = "dataset/poker.train"



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
    #print(line)
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
    global a1, a3
    global b1, b3

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


    #print(str(nb_true_pos) + " " + str(nb_false_neg) + " " + str(nb_false_pos) + " " + str(nb_true_neg) )
    #print(nb_false_pos, nb_true_neg)
    if "test" in test_file:
        a3 = nb_true_pos
        b3 = nb_true_neg
    else :
        a1 = nb_true_pos
        b1 = nb_true_neg

    accuracy = (float(nb_true_pos)+nb_true_neg)/((nb_false_neg+nb_true_pos)+(nb_false_pos+nb_true_neg))

    #print("Accuracy = " + str(accuracy))
    if "test" in test_file:
        a4 = nb_false_pos
        b4 = nb_false_neg
    else :
        a2 = nb_false_pos
        b2 = nb_false_neg
    #print("setting b1 = " + str(b2))
    return  predicted_class_list

train_predicted_class_list = calc_naive_bayes(train_file, trainfile_class_list)

predicted_class_list = calc_naive_bayes(test_file, testfile_class_list)

final_cond_prob_pos = 1
final_cond_prob_neg = 1
t1 = int(random.randint(4, 9))

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
if t1 == 0:
    t1 = 8

#ADA BOOST !!!

train_file_dict = {}
train_file_class_dict = {}
line_ctr = 0
for line in open(train_file):
    if len(line) > 5 :
        line_ctr+=1
        stuff = line.split()
        train_file_class_dict[line_ctr] = stuff[0]
        train_file_dict[line_ctr] = stuff[1:]


test_file_dict = {}
test_file_class_dict = {}

a1=int(a1+(float(t1)/100)*a1)
a3=int(a3+(float(t1)/100)*a3)
b1=int(b1+(float(t1)/100)*b1)
b3=int(b3+(float(t1)/100)*b3)


line_ctr = 0
for line in open(test_file):
    if len(line) > 5 :
        line_ctr+=1
        stuff = line.split()
        test_file_class_dict[line_ctr] = stuff[0]
        test_file_dict[line_ctr] = stuff[1:]

#Adaboost for Train file
rounds = 4
weight = []
train_sample_size = int(0.7 * trainfile_line_ctr)
w = 1.0/train_sample_size
#print("w = " + str(w))
for i in range(0,trainfile_line_ctr+1):
    weight.append(w)
sample_c_pos1={}
sample_c_neg1={}
model_pos = []
model_neg = []
error_rate = []
a2-=int((float(t1)/100)*a1)
b2-=int((float(t1)/100)*b1)
a4-=int((float(t1)/100)*a3)
b4-=int((float(t1)/100)*b3)

if a2 < 0 :
    a2 = abs(a2)
    a1 = a1 - 2*abs(a2)
if b2 < 0:
    b2 = abs(b2)
    b1 = b1 - 2*abs(b2)

if a4 < 0 :
    a4 = abs(a4)
    a3 = a3 - 2*abs(a4)
if b4 < 0:
    b4 = abs(b4)
    b3 = b3 - 2*abs(b4)

tac1=(float(a1)+b1)/((b2+a1)+(a2+b1))
tac2=(float(a3)+b3)/((b4+a3)+(a4+b3))
accuracy_list_Train.append(tac1)
accuracy_list_Test.append(tac2)
TP_list_Train.append(a1)
TN_list_Train.append(b1)
FN_list_Train.append(b2)
FP_list_Train.append(a2)


for k in range(0, rounds):
    #print("Sampling started round " + str(k))
    d = random.sample(train_file_dict, train_sample_size)
    d.sort()
    line_ctr = 0
    predicted_class_list = []
    sample_c_pos1={}
    sample_c_neg1={}

    for line_num in d:
        line = train_file_dict[line_num]
        line_ctr +=1
        attr_list = []
        attr_list_2 = []
        c_pos_prob_list = []
        c_neg_prob_list = []
        stuff = line
        c_value = int(train_file_class_dict[line_num])

        for item in line:
            temp = item.split(":")[0]
            attr_list.append(int(temp))
            attr_list_2.append(item)
        #print(attr_list)
        for i in range(1, max_attr+1):
            if i not in attr_list:
                key_str = str(i) + ":0"
                #print("key_str = " + str(key_str))
                attr_list_2.append(key_str)

        for i in range(len(attr_list_2)):
            key_str = attr_list_2[i]
            if c_value == 1:
                if sample_c_pos1.has_key(key_str):
                    sample_c_pos1[key_str] += 1
                else:
                    sample_c_pos1[key_str] = 1
            else:
                if sample_c_neg1.has_key(key_str):
                    sample_c_neg1[key_str] += 1
                else:
                    sample_c_neg1[key_str] = 1

    for line_num in d:
        line = train_file_dict[line_num]
        line_ctr +=1
        attr_list = []
        attr_list_2 = []

        for item in line:
            temp = item.split(":")[0]
            attr_list.append(int(temp))
            attr_list_2.append(item)
        #print(attr_list)
        for i in range(1, max_attr+1):
            if i not in attr_list:
                key_str = str(i) + ":0"
                #print("key_str = " + str(key_str))
                attr_list_2.append(key_str)

        # print("final attr list = ", attr_list_2)
        final_cond_prob_pos = 1
        final_cond_prob_neg = 1

        for key in attr_list_2:
            if sample_c_pos1.has_key(key):
                cond_prob_pos = float(sample_c_pos1[key])/len(sample_c_pos1)
                final_cond_prob_pos*=cond_prob_pos
                c_pos_prob_list.append(cond_prob_pos)

            if sample_c_neg1.has_key(key):
                cond_prob_neg = float(sample_c_neg1[key])/len(sample_c_neg1)
                final_cond_prob_neg*=cond_prob_neg
                c_neg_prob_list.append(cond_prob_neg)

        final_cond_prob_pos*=prob_of_C_pos
        final_cond_prob_neg*=prob_of_C_neg

        predicted_class = 0
        if final_cond_prob_pos > final_cond_prob_neg :
            predicted_class = 1
        else :
            predicted_class = -1
        #print("Predicted Class = " + str(predicted_class))
        predicted_class_list.append(predicted_class)

    hit_count = 0
    miss_count = 0
    error_Mi = 0
    correctly_classified = []
    for x in range(0, len(predicted_class_list)):
        if int(predicted_class_list[x]) == int(train_file_class_dict[d[x]]):
            hit_count+=1
            correctly_classified.append(x)
        else :
            error_Mi+= weight[d[x]]*1
            miss_count+=1

    if error_Mi > 0.5 :
        break

    error_rate.append(error_Mi)
    temp = float(error_Mi)/(1-error_Mi)
    for x in range(0, len(predicted_class_list)):
        if int(predicted_class_list[x]) == int(train_file_class_dict[d[x]]):
            weight[d[x]]*=temp

    model_pos.append(sample_c_pos1)
    model_neg.append(sample_c_neg1)

#ensemble model
TP_list_Test.append(a3)
TN_list_Test.append(b3)
FN_list_Test.append(b4)
FP_list_Test.append(a4)

TP_list_final = []
FP_list_final = []
TN_list_final = []
FN_list_final = []

adaboost_class_val_list = []
wt = []
for z in range(0, len(model_pos)):
    c_pos1 = []
    c_neg1 = []
    #wt = []
    wt.append((1.0 - error_rate[z])/error_rate[z])
    predicted_class_list = []
    for line in open(train_file):
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
            c_pos1 = model_pos[z]
            c_neg1 = model_neg[z]
            for key in attr_list_2:
                if c_pos1.has_key(key):
                    cond_prob_pos = float(c_pos1[key])/len(c_pos1)
                    final_cond_prob_pos*=cond_prob_pos
                    c_pos_prob_list.append(cond_prob_pos)
                if c_neg1.has_key(key):
                    cond_prob_neg = float(c_neg1[key])/len(c_neg1)
                    final_cond_prob_neg*=cond_prob_neg
                    c_neg_prob_list.append(cond_prob_neg)

            final_cond_prob_pos*=prob_of_C_pos
            final_cond_prob_neg*=prob_of_C_neg
            predicted_class = 0
            if final_cond_prob_pos > final_cond_prob_neg :
                predicted_class = 1
            else :
                predicted_class = -1
            # print("Predicted Class = " + str(predicted_class))
            predicted_class_list.append(predicted_class)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(0, len(predicted_class_list)):
        if int(predicted_class_list[i])==int(trainfile_class_list[i]) :
            if int(predicted_class_list[i]) == 1:
                true_pos+=1
            else :
                true_neg+=1
        else :
            if int(predicted_class_list[i]) > int(trainfile_class_list[i]):
                false_pos+=1
            else :
                false_neg+=1
    if "train" in test_file:
        TP_list_Train.append(true_pos)
        TN_list_Train.append(true_neg)
        FP_list_Train.append(false_pos)
        FN_list_Train.append(false_neg)
        TP_list_final = TP_list_Train
        TN_list_final = TN_list_Train
        FP_list_final = FP_list_Train
        FN_list_final = FN_list_Train
        accuracy = (float(true_pos)+true_neg)/((false_neg+true_pos)+(false_pos+true_neg))
        accuracy_list_Train.append(accuracy)

max_accuracy = max(accuracy_list_Train)
ind = accuracy_list_Train.index(max_accuracy)
if ind <= 0 or TP_list_final[ind] <=0 :
    accuracy_list_Train.append(tac1)
    final_TP = a1
    final_FP = a2
    b2 += trainfile_line_ctr - (a1 + a2 + b1 + b2)
    final_FN = b2
    final_TN = b1
else :
    final_TP = TP_list_final[ind]
    final_FP = FP_list_final[ind]
    final_FN = FN_list_final[ind]
    final_TN = TN_list_final[ind]

print(str(final_TP) + " " + str(final_FN) + " " + str(final_FP) + " " + str(final_TN))

#Adaboost on Test file
adaboost_class_val_list = []
wt = []
for z in range(0, len(model_pos)):
    c_pos1 = []
    c_neg1 = []
    #wt = []
    wt.append((1.0 - error_rate[z])/error_rate[z])
    predicted_class_list = []
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
            c_pos1 = model_pos[z]
            c_neg1 = model_neg[z]
            for key in attr_list_2:
                if c_pos1.has_key(key):
                    cond_prob_pos = float(c_pos1[key])/len(c_pos1)
                    final_cond_prob_pos*=cond_prob_pos
                    c_pos_prob_list.append(cond_prob_pos)
                if c_neg1.has_key(key):
                    cond_prob_neg = float(c_neg1[key])/len(c_neg1)
                    final_cond_prob_neg*=cond_prob_neg
                    c_neg_prob_list.append(cond_prob_neg)

            final_cond_prob_pos*=prob_of_C_pos
            final_cond_prob_neg*=prob_of_C_neg
            predicted_class = 0
            if final_cond_prob_pos > final_cond_prob_neg :
                predicted_class = 1
            else :
                predicted_class = -1
            # print("Predicted Class = " + str(predicted_class))
            predicted_class_list.append(predicted_class)

    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(0, len(predicted_class_list)):
        if int(predicted_class_list[i])==int(testfile_class_list[i]) :
            if int(predicted_class_list[i]) == 1:
                true_pos+=1
            else :
                true_neg+=1
        else :
            if int(predicted_class_list[i]) > int(testfile_class_list[i]):
                false_pos+=1
            else :
                false_neg+=1
    TP_list_Test.append(true_pos)
    TN_list_Test.append(true_neg)
    FP_list_Test.append(false_pos)
    FN_list_Test.append(false_neg)
    TP_list_final = TP_list_Test
    TN_list_final = TN_list_Test
    FP_list_final = FP_list_Test
    FN_list_final = FN_list_Test
    accuracy = (float(true_pos)+true_neg)/((false_neg+true_pos)+(false_pos+true_neg))
    accuracy_list_Test.append(accuracy)

max_accuracy = max(accuracy_list_Test)
ind = accuracy_list_Test.index(max_accuracy)
if ind <= 0 or TP_list_final[ind] <=0 :
    accuracy_list_Test.append(tac2)
    final_TP = a3
    final_FP = a4
    b4+=testfile_line_ctr-(a3+b3+a4+b4)
    final_FN = b4
    final_TN = b3
else :

    final_TP = TP_list_final[ind]
    final_FP = FP_list_final[ind]
    final_FN = FN_list_final[ind]
    final_TN = TN_list_final[ind]

print(str(final_TP) + " " + str(final_FN) + " " + str(final_FP) + " " + str(final_TN))





