"""
import
"""

# std
import csv
import numpy as np
import time
import matplotlib.pyplot as plt

# sklearn
from sklearn import tree
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

"""
global
"""

class Classifier:
  def __init__(self, 
               files, 
               delim=';', 
               boundaries = [5, 7], 
               labels = [0, 1, 2]):
    self.delim = delim
    self.files = files
    self.boundaries = boundaries
    self.labels = labels 
    self.verbose = False
    self.x = []
    self.y = []

  def setup_features(self):
    '''
    setup i/o
    '''
    self.Features = np.ones((len(self.x[0]),len(self.x)))

    for i in range(len(self.x)):
      self.Features[:,i] = self.x[i]

  def plot(self, title, xtitle, ytitle, x, y):
    plt.plot(x, y)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()

  def setup_classifier(self):
    self.y = np.array(self.y)
    self.y = np.reshape(self.y,(self.y.size,1))
    self.ylabel = np.ones((self.y.shape))

    for i in range(self.y.size):
      self.ylabel[i] = self.labels[-1]
      for j in range(len(self.boundaries)):
        if self.y[i] < j:
          print(i, j)
          self.ylabel[i] = self.labels[j]

  def read_file(self, csvname):
    '''
    This function reads the winequality datasets and
    appends the results into the appropriate feature
    and output arrays
    '''
    with open(csvname) as f:
      reader = csv.DictReader(f, delimiter=self.delim)
      num_fields = len(reader.fieldnames)
      if(len(self.x) == 0):
        for i in range(0, num_fields - 1):
          self.x.append([])
      for row in reader:
        r = list(csv.OrderedDict(row).values())
        for i in range(num_fields - 1):
          self.x[i].append(float(r[i]))
        self.y.append(float(r[num_fields-1]))
    return

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
      testx_temp = np.zeros((int(round(x.shape[0]/k))+1,x.shape[1]))
      testy_temp = np.zeros((int(round(x.shape[0]/k))+1,y.shape[1]))
      trainx_temp = np.zeros((int(((k-1)*round(x.shape[0]/k)))+1,x.shape[1]))
      trainy_temp = np.zeros((int(((k-1)*round(x.shape[0]/k)))+1,y.shape[1]))
      l=int((i.size/k)*(h+1))
      count = 0
      for j in range(0,i.size):
        if ((j)+int(i.size/k)*(h) >= l):
          trainx_temp[count] = (np.take(xtemp,i[j],
                                        axis=0))
          trainy_temp[count] = (np.take(ytemp,i[j]))
          count+=1
        else:
          testx_temp[j] = (np.take(xtemp,i[(j)+
                           int(i.size/k)*(h)],axis=0))
          testy_temp[j] = (np.take(ytemp,i[(j)+
                           int(i.size/k)*(h)]))
      # train x, y
      trainx[h] = trainx_temp
      trainy[h] = trainy_temp

      # test x, y
      testx[h] = testx_temp
      testy[h] = testy_temp

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
    if(self.verbose):
      print("training time = "+ str(train_time))    
      
    stime = time.time()
    prediction = algorithm.predict(testx)
    etime = time.time()
    train_time = etime-stime
    if(self.verbose):
      print("testing time = "+ str(train_time))
    
    accuracy = accuracy_score(testy,prediction)
    if(self.verbose):
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

    print("-----", 
          best_mod.__class__.__name__,
          "-----")
    print("Accuracy: " + str(best_acc))
    etime = time.time()
    tot_time = etime-stime
    print("Time:", str(tot_time))
    print("----------")
    print()

    return best_mod,best_acc,tot_time

  def test_models(self, 
                  Features, 
                  trainx, 
                  trainy, 
                  testx, 
                  testy, 
                  ylabel):
    models = []
    accs = []
    times = []
    list_models = [GaussianNB(),
                   LogisticRegression(),
                   tree.DecisionTreeClassifier(),
                   RandomForestClassifier(n_estimators=1000),
                   KNeighborsClassifier(),
                   MLPClassifier(alpha = 1)]  

    for i in list_models:
      mod_i, acc_i,times_i = self.bestK_retrain(self.Features, 
                            i,
                            trainx,
                            trainy,
                            testx,
                            testy, 
                            ylabel)
      models.append(mod_i)
      accs.append(acc_i)
      times.append(times_i)
    return models,accs,times

  def load(self):
    for file in self.files:
      self.read_file(file)

  def train(self):
    # k_fold
    self.trainx,self.trainy,self.testx,self.testy = self.k_fold_split(self.Features, self.ylabel)

    #reshape for sklearn
    self.testy = np.reshape(self.testy,(self.testy.shape[0],
                              self.testy.shape[1],))
    self.trainy = np.reshape(self.trainy,(self.trainy.shape[0],
                                self.trainy.shape[1],))

    # output
    print("-----")
    print("testx shape: " +
           str(self.testx.shape))
    print("testy shape: " +
           str(self.testy.shape))
    print("trainx shape: " +
           str(self.trainx.shape))
    print("trainy shape: " +
           str(self.trainy.shape))
    print("-----")
    print()

    #self.plot("title", "xtitle", "ytitle", 
               #self.testx, self.testy)

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
    print("=====")
    print("Most accurate: " +
           self.models[self.acc_ind].__class__.__name__)
    print("Fastest: "+         self.models[self.time_ind].__class__.__name__)
    print("=====")

  def run(self):
    self.load()
    self.setup_features()
    self.setup_classifier()
    self.train()
    self.compare()
    self.results()

"""
main
"""
def main():
  c = Classifier(["winequality-red.csv",
                  "winequality-white.csv"])
  c.run()
"""
run
"""
main()
