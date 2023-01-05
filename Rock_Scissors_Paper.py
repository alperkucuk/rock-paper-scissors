import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy import ndimage as nd
from skimage.filters import sobel

# Data directory and data category 
DIRECTORY = r"D:/Datasets/Rock-Paper-Scissors/train"
CATEGORIES = os.listdir(DIRECTORY)
df = pd.DataFrame( columns = ['Entropy', 'Gaussian', 'Sobel'])

data = []
labels = []
features = []
# get all the data and labels in given directory
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    filelist = os.listdir(path)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    
    for img in filelist:
        img_path = os.path.join(path, img)
        image = load_img(img_path)
        
        image = img_to_array(image)
        image = preprocess_input(image)
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img2 = img.reshape(-1)


        entropy_img = entropy(img, disk(1))

        entropy1 = entropy_img.reshape(-1).mean()
        features.append(entropy1) 

        

       

        gaussian_img = nd.gaussian_filter(img, sigma=3)

        gaussian_img1 = gaussian_img.reshape(-1).mean()
        features.append(gaussian_img1) 

        sobel_img = sobel(img)

        sobel1 = sobel_img.reshape(-1).mean()
        features.append(sobel1) 
        
        
        df.loc[len(df.index)] = features
        
        
        features = []
        
        labels.append(category)


cv2.imshow('entropy', entropy_img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('gaussian', gaussian_img)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('sobel', sobel_img)
cv2.waitKey()
cv2.destroyAllWindows()

print(df)

import collections

import collections

# check the data count and balance  
counter=collections.Counter(labels)
print(counter)

X = df
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

ks = [1,3,5,7,10]

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    scores = cross_val_score(
    knn, X_train, y_train, cv=10, scoring='accuracy')
    
    print('KNN K={} ile Train Başarısı {}'.format(k,scores.mean()*100))
    
    scores = cross_val_score(
    knn, X_test, y_test, cv=10, scoring='accuracy')
    
    print('KNN K={} ile Test Başarısı {}'.format(k,scores.mean()*100))
    
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    clf = SVC(kernel=kernel)
    scores = cross_val_score(
        clf, X_train, y_train, cv=10, scoring='accuracy')
    print('SVM Kernel={} ile train Başarısı {}'.format(kernel,scores.mean()*100))
    
    scores = cross_val_score(
        clf, X_test, y_test, cv=10, scoring='accuracy')    
    
    print('SVM Kernel={} ile test Başarısı {}'.format(kernel,scores.mean()*100))

    
# Applying Decision Tree
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
    

scores_train = cross_val_score(
    clf, X_train, y_train, cv=10, scoring='accuracy')

scores_test = cross_val_score(
    clf, X_test, y_test, cv=10, scoring='accuracy')

print('Decision Train Tree Başarısı = {}'.format((scores_train.mean()*100)))
print('Decision Test Tree Başarısı = {}'.format((scores_test.mean()*100)))

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train)

scores_train = cross_val_score(
    knn, X_train, y_train, cv=10, scoring='accuracy')

scores_test = cross_val_score(
    knn, X_test, y_test, cv=10, scoring='accuracy')

print('KNN Train Başarısı = {}'.format((scores_train.mean()*100)))
print('KNN Test Başarısı = {}'.format((scores_test.mean()*100)))

import numpy as np
import matplotlib.pyplot as plt

class_names = ['Rock', 'Scissors', 'Paper']

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        knn,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from scipy import ndimage as nd
from skimage.filters import sobel

# Data directory and data category 
DIRECTORY = r"D:/Datasets/Rock-Paper-Scissors/pred"
CATEGORIES = os.listdir(DIRECTORY)
df = pd.DataFrame( columns = ['Entropy', 'Gaussian', 'Sobel'])

data = []
features = []
# get all the data and labels in given directory
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    filelist = os.listdir(path)
    for fichier in filelist[:]: # filelist[:] makes a copy of filelist.
        if not(fichier.endswith(".png")):
            filelist.remove(fichier)
    
    for img in filelist:
        img_path = os.path.join(path, img)
        image = load_img(img_path)
        
        image = img_to_array(image)
        image = preprocess_input(image)
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img2 = img.reshape(-1)


        entropy_img = entropy(img, disk(1))

        entropy1 = entropy_img.reshape(-1).mean()
        features.append(entropy1) 

        

       

        gaussian_img = nd.gaussian_filter(img, sigma=3)

        gaussian_img1 = gaussian_img.reshape(-1).mean()
        features.append(gaussian_img1) 

        sobel_img = sobel(img)

        sobel1 = sobel_img.reshape(-1).mean()
        features.append(sobel1) 
        
        
        df.loc[len(df.index)] = features
        
        
        features = []


clf.predict(df)


