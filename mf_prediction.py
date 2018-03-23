#from sklearn import tree 
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# [height, weight, shoe_size]

X=np.array([[180, 80, 43],[176,69,44], [161,60,39],[155,55,38],[167,65,41],[191,91,48],[175,64,40],[178,71,41],[160,56,38],[172,76,42],[181,86,43]])

Y= np.array(['male','male','female','female','male','male','female','female','female','male','male'])

#clf= tree.DecisionTreeClassifier()

#clf= RandomForestClassifier(max_depth =2,random_state=0)

clf= SVC()
#train data

clf= clf.fit(X,Y)


prediction = clf.predict([[161,60,39]])

print(prediction)
