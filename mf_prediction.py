from sklearn import tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
# [height, weight, shoe_size]

X=np.array([[180, 80, 43],[176,69,44], [161,60,39],[155,55,38],[167,65,41],[191,91,48],[175,64,40],[178,71,41],[160,56,38],[172,76,42],[181,86,43]])

Y= np.array(['male','male','female','female','male','male','female','female','female','male','male'])

clf_tree= tree.DecisionTreeClassifier()
clf_RFC= RandomForestClassifier(max_depth =2,random_state=0)
clf_svm= SVC()
#train data

clf_tree= clf_tree.fit(X,Y)
clf_RFC= clf_RFC.fit(X,Y)
clf_svm= clf_svm.fit(X,Y)

#print the prediction

pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for Decisiontree: {}'.format(acc_tree))


pred_RFC = clf_tree.predict(X)
acc_RFC = accuracy_score(Y, pred_RFC) * 100
print('Accuracy for RFC: {}'.format(acc_RFC))


pred_svm = clf_tree.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

#Camparisiion of classifier

index= np.argmax([acc_tree, acc_RFC, acc_svm])

classifiers = {0:'DT',1:'RFC',2:'SVM'}
print('classifier with highest accuracy:'.format(classifiers[index]))
