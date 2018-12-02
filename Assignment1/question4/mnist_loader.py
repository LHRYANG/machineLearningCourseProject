#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def ReadData(filename):
    file = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(file,encoding="latin-1")
    file.close()
    return training_data, validation_data, test_data

def ChangeData(filename):

    training_data, validation_data, test_data = ReadData(filename)

    training_inputs = [np.reshape(x, 784) for x in training_data[0]]

    #training_outputs = [change(y) for y in training_data[1]]

    training_data = list(zip(training_inputs, training_data[1]))
    validation_inputs = [np.reshape(x, 784) for x in validation_data[0]]

    validation_data = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, 784) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return training_data, validation_data, test_data

def change(j):

    result = np.zeros(19)
    result[j] = 1.0
    return result

def constructData(filename):
    print("start to construct train and test:")
    training_data,validation_data,test_data=ChangeData(filename)

    train_len=len(training_data)
    test_len=len(test_data)

    train=[]
    test=[]
    train_size=100000
    test_size=10000
    for i in range(train_size):
        index=np.random.randint(train_len,size=2)

        train_input=np.append(training_data[index[0]][0],training_data[index[1]][0])

        train_output=change(training_data[index[0]][1]+training_data[index[1]][1])

        train_one=[train_input,train_output]
        train.append(train_one)

    for i in range(test_size):
        index = np.random.randint(test_len, size=2)

        test_input = np.append(test_data[index[0]][0], test_data[index[1]][0])

        test_output = change(test_data[index[0]][1] + test_data[index[1]][1])

        test_one = [test_input, test_output]
        test.append(test_one)

    with open("./data/train.pkl","wb") as f:
        pickle.dump(train,f)
    with open("./data/test.pkl","wb") as f:
        pickle.dump(test,f)

constructData("./data/mnist.pkl.gz")


