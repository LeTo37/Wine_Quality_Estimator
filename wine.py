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

x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x10 = []
x11 = []
y = []

"""
function
"""
def read_file(csvname):
  '''
  This function reads the winequality datasets and 
  appends the results into the appropriate feature 
  and output arrays
  '''
  with open(csvname) as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
      x1.append(float(row['fixed acidity']))
      x2.append(float(row['volatile acidity']))
      x3.append(float(row['citric acid']))
      x4.append(float(row['residual sugar']))
      x5.append(float(row['chlorides']))
      x6.append(float(row['free sulfur dioxide']))
      x7.append(float(row['total sulfur dioxide']))
      x8.append(float(row['density']))
      x9.append(float(row['pH']))
      x10.append(float(row['sulphates']))
      x11.append(float(row['alcohol']))
      y.append(float(row['quality']))

def k_fold_split(x,y,k=3):
  '''
  This function takes the full data set of features   and labels/output and splits the set into k=3 
  sections
  2/3 is used for training 
  1/3 for testing. 
  It rotates these folds such each set is used for 
  both training and testing
  '''
  testx = np.zeros((k,int(round(x.shape[0]/k))+1,x.shape[1])) # +1 because x.shape[0]/3 is not whole
  testy = np.zeros((k,int(round(x.shape[0]/k))+1,y.shape[1]))
  trainx = np.zeros((k,int(((k-1)*round(x.shape[0]/k)))+1,x.shape[1]))
  trainy = np.zeros((k,int(((k-1)*round(x.shape[0]/k)))+1,y.shape[1]))
  i = np.random.choice(range(x.shape[0]),x.shape[0],replace = False)
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
        testx_temp[j] = (np.take(xtemp,i[(j)+int(i.size/k)*(h)],axis=0))
        testy_temp[j] = (np.take(ytemp,i[(j)+int(i.size/k)*(h)]))
      else:
        trainx_temp[count] = (np.take(xtemp,i[j],axis=0))
        trainy_temp[count] = (np.take(ytemp,i[j]))
        count+=1
    testx[h] = testx_temp
    testy[h] = testy_temp
    trainx[h] = trainx_temp
    trainy[h] = trainy_temp
  return trainx,trainy,testx,testy

def train_and_test(model,trainx,trainy,testx,testy):
  stime = time.time()
  algorithm = model.fit(trainx,trainy)
  etime = time.time()
  train_time = etime-stime
  log = True
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
    print("The accuracy of this prediction is: " + str(accuracy))
  return algorithm,accuracy

def bestK_retrain(Features, model,trainx,trainy,testx,testy, ylabel):
  stime = time.time()
  mod = []
  acc = []
  for k in range(0,trainx.shape[0]):
    mod_k,acc_k = train_and_test(model,trainx[k],trainy[k],testx[k],testy[k])
    mod.append(mod_k)
    acc.append(acc_k)
  acc = np.array(acc)
  ind = np.argmax(acc)
  best_mod = mod[ind]
  best_pred = best_mod.predict(Features)
  best_acc = accuracy_score(ylabel,best_pred)
  print("The best accuracy achieved for "+ best_mod.__class__.__name__+ " is: " + str(best_acc))
  etime = time.time()
  tot_time = etime-stime
  print("Time taken for "+ best_mod.__class__.__name__+ " model= "+ str(tot_time))
  return best_mod,best_acc,tot_time

def test_models(Features, trainx, trainy, testx, testy, ylabel):
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
    mod_i, acc_i,times_i = bestK_retrain(Features, i,trainx,trainy,testx,testy, ylabel)
    print("")
    models.append(mod_i)
    accs.append(acc_i)
    times.append(times_i)
    
  return models,accs,times

"""
main
"""
def main():
  global x1, x2, x3, x4, x5, x6, x7, x8, x9, x19, x11
  global y

  # load
  read_file("winequality-red.csv")
  read_file("winequality-white.csv")

  # setup i/o
  Features = np.ones((len(x1),11))
  y = np.array(y)
  y = np.reshape(y,(y.size,1))
  Features[:,0] = x1
  Features[:,1] = x2
  Features[:,2] = x3
  Features[:,3] = x4
  Features[:,4] = x5
  Features[:,5] = x6
  Features[:,6] = x7
  Features[:,7] = x8
  Features[:,8] = x9
  Features[:,9] = x10
  Features[:,10] = x11

  ylabel = np.ones((y.shape))
  for i in range(0,y.size):
    if (y[i] < 5):
      ylabel[i] = 0
    elif (y[i] < 7):
      ylabel[i] = 1
    else:
      ylabel[i] = 2

  # train
  trainx,trainy,testx,testy = k_fold_split(Features,ylabel)
  #reshape for sklearn
  testy = np.reshape(testy,(testy.shape[0],testy.shape[1],))
  trainy = np.reshape(trainy,(trainy.shape[0],trainy.shape[1],))
  print("testx has the following shape: " + str(testx.shape))
  print("testy has the following shape: " + str(testy.shape))
  print("trainx has the following shape: " + str(trainx.shape))
  print("trainy has the following shape: " + str(trainy.shape))

  # compare

  models,accs,times = test_models(Features, trainx, trainy, testx, testy, ylabel)
  models = np.array(models)
  accs = np.array(accs)
  times = np.array(times)
  acc_ind = np.argmax(accs)
  time_ind = np.argmin(times)
  print("The model with the best accuray is " + models[acc_ind].__class__.__name__)
  print("The model with the fastest processing is " + models[time_ind].__class__.__name__)

"""
run
"""
main()
