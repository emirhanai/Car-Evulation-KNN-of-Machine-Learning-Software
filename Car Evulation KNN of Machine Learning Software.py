import numpy as np
from sklearn import preprocessing, neighbors, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd




#read to the csv in data :)
df = pd.read_csv('car-evulation-domain.csv')

X = df.drop(['class'], axis = 'columns')
y = df.drop(['lug-boot','buying','maint','doors','persons','safety'],axis='columns')

#random id to data in column one
yx_change = LabelEncoder()

#change columns of y :)
y['class_Change'] = yx_change.fit_transform(y['class'])

#changed columns in new y >> y_n
y_n = y.drop(['class'], axis= 'columns')

#change columns of X :)
X['buying_Change'] = yx_change.fit_transform(X['buying'])
X['maint_Change'] = yx_change.fit_transform(X['maint'])
X['safety_Change'] = yx_change.fit_transform(X['safety'])
X['lug-boot_Change'] = yx_change.fit_transform(X['lug-boot'])
X['doors_Change'] = yx_change.fit_transform(X['doors'])
X['persons_Change'] = yx_change.fit_transform(X['persons'])

#changed of columns in new X >> X_n
X_n = X.drop(['buying','maint','safety','lug-boot','doors','persons'], axis='columns')


print(y_n)
print('\n\n')
print(X_n)

#X train and test, y train and test creating with model_selection and train_test_split from X_n and y_n.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_n,y_n,test_size = 0.3)

#neighbors of KNN
neighbors_clf = neighbors.KNeighborsClassifier()

neighbors_clf.fit(X_train,y_train)

accuracy = neighbors_clf.score(X_test,y_test)

#accuracy the print.
print("Accuracy of the: ",accuracy)

predict_buying = int(input("You are entering to 'buying': "))
predict_maint = int(input("You are entering to 'maint': "))
predict_doors = int(input("You are entering to 'doors': "))
predict_persons = int(input("You are entering to 'persons': "))
predict_lug_boot = int(input("You are entering to 'persons': "))
predict_safety = int(input("You are entering to 'safety': "))



#predict_quest = np.array([[predict_buying,predict_maint,predict_doors,predict_persons,predict_lug_boot,predict_safety]])

#prediction = neighbors_clf.predict(predict_quest)

#prediction with model in data
#print("Prediction Class: ",prediction)

try:
    while True:
        model_run = neighbors_clf.predict(np.array([[predict_buying,predict_maint,predict_doors,predict_persons,predict_lug_boot,predict_safety]]))
        countrys = pd.read_csv('class_name.csv',index_col=None, na_values=None)
        countrys_detect_algorithm = countrys.columns.values[model_run]
        print("Predicted class: {}".format(countrys_detect_algorithm))
        break
except:
    print("Repeat Try :))")


#unacc = 2

#vgood = 3

#good = 1

#acc = 0