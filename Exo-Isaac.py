
#Import des bibliothèques
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import io
import requests 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#Chargement de la DataSet
dataset= pd.read_excel("data_evenement_cardiaque.xlsx")
print(dataset)
print(dataset.shape) #Pour connaitre les dimensions de notre dataset
print(dataset.head) #Pour voir les informations qui sont en tete de la dataset
print(dataset.describe())# Afficher les statistiques de la dataset

#On determine les variables indépendantes X
X=dataset[['age','baseef','dobef','possef','restwma','event']]
X=X.values
#On determine les variables dépendantes inddependantes X et y
X=dataset[['age','baseef','dobef','possef','restwma']]
y=dataset[['event']]
print(X.head())
print(y.head())

corr= np.corrcoef(dataset)
print(corr)
plt.imshow(np.corrcoef(X.T),cmap='Blues')# La correlation entre les colonnesc-a-d entre les variables
plt.colorbar()
plt.show

#On divise l'ensemble des données en train et test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)    
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape) 

#On cree la prediction du model
model= LogisticRegression()
print(model.fit(x_train,y_train))
y_prediction= model.predict(x_test)
print(y_prediction)
print(model.score(x_train, y_train))

#On imprime le rapport de classification
print("prediction : {}".format(accuracy_score(y_test,y_prediction)*100))
print(classification_report(y_test,y_prediction))

#plotting confusion matrix on heatmap
confusion_matrix=confusion_matrix(y_test,y_prediction)
#sns.heatmap(confusion_matrix, annot=True,xticklabels=['not_event','event'], yticklabels=['not_event','event'])
#plt.figure(figsize=(4,4))
#plt.show()

#Verification
print(x_test.head())
print(y_test.head())
print(y_prediction[:4])