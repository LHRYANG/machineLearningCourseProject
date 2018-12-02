import pickle
import numpy as np
class dataLoader:

    def __init__(self,train_path,test_path):
        with open(train_path,"rb") as f:
            self.train=pickle.load(f)
        with open(test_path, "rb") as f:
            self.test=pickle.load(f)
        self.test_image=[self.test[i][0] for i in range(len(self.test))]
        self.test_label=[self.test[i][1] for i in range(len(self.test))]
        self.test_image=np.array(self.test_image)
        self.test_label=np.array(self.test_label)

        self.trainlen=len(self.train)
    def nextBatch(self,batch_size):
        batch=[]
        indexs=np.random.randint(self.trainlen,size=batch_size)
        train_image=[]
        train_label=[]
        for index in indexs:
            train_image.append(self.train[index][0])
            train_label.append(self.train[index][1])
        train_image=np.array(train_image)
        train_label=np.array(train_label)

        return train_image,train_label






