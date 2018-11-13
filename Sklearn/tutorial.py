from sklearn import datasets,svm
import xgboost as xgb
import pickle

iris=datasets.load_iris()
digits=datasets.load_digits()

clf=svm.SVC(gamma=0.001,C=100.)
X,y=iris.data,iris.target
clf.fit(X,y)

s=pickle.dumps(clf)
clf2=pickle.loads(s)



print(clf2.predict(X[-1:]))