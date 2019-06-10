"""
import
"""
import csv
import numpy as np
import time
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

"""
global
"""

class Classifier:
  def __init__(self):
    self.Features = self.x1 = []
    self.Features = self.x2 = []
    self.Features = self.x3 = []
    self.x4 = []
    self.x5 = []
    self.x6 = []
    self.x7 = []
    self.x8 = []
    self.x9 = []
    self.x10 = []
    self.x11 = []
    self.y = []

  def setup_features(self):
    '''
    setup i/o
    '''
    self.Features = np.ones((len(self.x1),11))
    self.Features[:,0] = self.x1
    self.Features[:,1] = self.x2
    self.Features[:,2] = self.x3
    self.Features[:,3] = self.x4
    self.Features[:,4] = self.x5
    self.Features[:,5] = self.x6
    self.Features[:,6] = self.x7
    self.Features[:,7] = self.x8
    self.Features[:,8] = self.x9
    self.Features[:,9] = self.x10
    self.Features[:,10] = self.x11

  def setup_classifiers(self):
    self.y = np.array(self.y)
    self.y = np.reshape(self.y,(self.y.size,1))
    self.ylabel = np.ones((self.y.shape))
    for i in range(0,self.y.size):
      if (self.y[i] < 5):
        self.ylabel[i] = 0
      elif (self.y[i] < 7):
        self.ylabel[i] = 1
      else:
        self.ylabel[i] = 2

  def read_file(self, csvname):
    '''
    This function reads the winequality datasets and
    appends the results into the appropriate feature
    and output arrays
    '''
    with open(csvname) as f:
      reader = csv.DictReader(f, delimiter=';')
      for row in reader:
        self.x1.append(float(row['fixed acidity']))
        self.x2.append(float(row['volatile acidity']))
        self.x3.append(float(row['citric acid']))
        self.x4.append(float(row['residual sugar']))
        self.x5.append(float(row['chlorides']))
        self.x6.append(float(row['free sulfur dioxide']))
        self.x7.append(float(row['total sulfur dioxide']))
        self.x8.append(float(row['density']))
        self.x9.append(float(row['pH']))
        self.x10.append(float(row['sulphates']))
        self.x11.append(float(row['alcohol']))
        self.y.append(float(row['quality']))

  def k_fold_split(self, x, y, k=3):
    '''
    This function takes the full data set of 
    features and labels/output and splits the set 
    into k=3 sections
    2/3 is used for training 
    1/3 for testing. 
    It rotates these folds such each set is used for
    both training and testing
    '''
    testx = np.zeros((k,int(round(x.shape[0]/k))+1,x.shape[1])) # +1 because x.shape[0]/3 is not whole
    testy = np.zeros((k,int(round(x.shape[0]/k))+1,y.shape[1]))
    trainx = np.zeros((k,int(((k-1)*round(x.shape[0]/k)))+1,x.shape[1]))
    trainy = np.zeros((k,int(((k-1)*round(x.shape[0]/k)))+1,y.shape[1]))
    i = np.random.choice(range(x.shape[0]),
                         x.shape[0],
                         replace = False)
    for h in range(0,k):
      xtemp = np.copy(x)
      ytemp = np.copy(y)
      print(x.shape[0],k, x.shape[0]/k)
      testx_temp = np.zeros((int(round(x.shape[0]/k))+1,x.shape[1]))
      testy_temp = np.zeros((int(round(x.shape[0]/k))+1,y.shape[1]))
      trainx_temp = np.zeros((int(((k-1)*round(x.shape[0]/k)))+1,x.shape[1]))
      trainy_temp = np.zeros((int(((k-1)*round(x.shape[0]/k)))+1,y.shape[1]))
      l=int((i.size/k)*(h+1))
      count = 0
      for j in range(0,i.size):
        if ((j)+int(i.size/k)*(h) < l):
          testx_temp[j] = (np.take(xtemp,i[(j)+
                           int(i.size/k)*(h)],axis=0))
          testy_temp[j] = (np.take(ytemp,i[(j)+
                           int(i.size/k)*(h)]))
        else:
          trainx_temp[count] = (np.take(xtemp,i[j],
                                        axis=0))
          trainy_temp[count] = (np.take(ytemp,i[j]))
          count+=1
      testx[h] = testx_temp
      testy[h] = testy_temp
      trainx[h] = trainx_temp
      trainy[h] = trainy_temp
    return trainx, trainy, testx, testy

  def train_and_test(self, 
                     model, 
                     trainx, 
                     trainy, 
                     testx, 
                     testy):
    stime = time.time()
    algorithm = model.fit(trainx, trainy)
    etime = time.time()
    train_time = etime-stime
    log = False
    if(log):
      print("training time = "+ str(train_time))    
      
    stime = time.time()
    prediction = algorithm.predict(testx)
    etime = time.time()
    train_time = etime-stime
    if(log):
      print("testing time = "+ str(train_time))
    
    accuracy = accuracy_score(testy,prediction)
    if(log):
      print("The accuracy of this prediction is: " + 
             str(accuracy))
    return algorithm, accuracy

  def bestK_retrain(self, 
                    Features, 
                    model,
                    trainx,
                    trainy,
                    testx, 
                    testy, 
                    ylabel):
    stime = time.time()
    mod = []
    acc = []
    for k in range(0,trainx.shape[0]):
      mod_k,acc_k = self.train_and_test(model,trainx[k],trainy[k],testx[k],testy[k])
      mod.append(mod_k)
      acc.append(acc_k)
    acc = np.array(acc)
    ind = np.argmax(acc)
    best_mod = mod[ind]
    best_pred = best_mod.predict(self.Features)
    best_acc = accuracy_score(ylabel,best_pred)
    print("The best accuracy achieved for "+ best_mod.__class__.__name__+ " is: " + str(best_acc))
    etime = time.time()
    tot_time = etime-stime
    print("Time: "+ best_mod.__class__.__name__+ " model= "+ str(tot_time))
    return best_mod,best_acc,tot_time

  def test_models(self, Features, trainx, trainy, testx, testy, ylabel):
    Gaussian_model= GaussianNB()
    LogReg_model = LogisticRegression()
    DTree_model = tree.DecisionTreeClassifier()
    RForest_model = RandomForestClassifier(n_estimators=1000)
    KNN_model = KNeighborsClassifier()
    Net_model = MLPClassifier(alpha = 1)
    list_models = [Gaussian_model,LogReg_model,DTree_model,RForest_model,KNN_model,Net_model]  
    models = []
    accs = []
    times = []
    for i in list_models:
      mod_i, acc_i,times_i = self.bestK_retrain(self.Features, i,trainx,trainy,testx,testy, ylabel)
      print("")
      models.append(mod_i)
      accs.append(acc_i)
      times.append(times_i)
    return models,accs,times

  def load(self):
    self.read_file("winequality-red.csv")
    self.read_file("winequality-white.csv")

  def train(self):
    self.trainx,self.trainy,self.testx,self.testy = self.k_fold_split(self.Features, self.ylabel)
    #reshape for sklearn
    self.testy = np.reshape(self.testy,(self.testy.shape[0],
                              self.testy.shape[1],))
    self.trainy = np.reshape(self.trainy,(self.trainy.shape[0],
                                self.trainy.shape[1],))
    print("testx has the following shape: " +
           str(self.testx.shape))
    print("testy has the following shape: " +
           str(self.testy.shape))
    print("trainx has the following shape: " +
           str(self.trainx.shape))
    print("trainy has the following shape: " +
           str(self.trainy.shape))

  def compare(self):
    models,accs,times = self.test_models(self.Features,
                                         self.trainx,
                                         self.trainy,
                                         self.testx,
                                         self.testy,
                                         self.ylabel)
    self.models = np.array(models)
    self.accs = np.array(accs)
    self.times = np.array(times)
    self.acc_ind = np.argmax(accs)
    self.time_ind = np.argmin(times)

  def results(self):
    print("The model with the best accuray is " +
           self.models[self.acc_ind].__class__.__name__)
    print("The model with the fastest processing is "+         self.models[self.time_ind].__class__.__name__)

  def run(self):
    self.load()
    self.setup_features()
    self.setup_classifiers()
    self.train()
    self.compare()
    self.results()

"""
main
"""
def main():
  c = Classifier()
  c.run()

"""
run
"""
main()
