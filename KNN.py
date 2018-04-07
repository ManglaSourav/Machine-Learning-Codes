import  numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True) #replace missing(?) values by -99999(most algorithm recognise thats as an outlier)
    df.drop(['id'],1, inplace=True) #drop id column (no need of that)

    x = np.array(df.drop(['class'], 1)) #features are everything except class(output) column
    y = np.array(df['class'])           #only class column

    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2) #Split arrays or matrices into random train and test subsets

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    # print(accuracy) #confidance
    # example_measures = np.array([[4,2,1,1,1,2,3,2,1], [1,2,3,4,5,6,7,8,9]])
    # example_measures = example_measures.reshape(len(example_measures),-1)
    #
    # prediction = clf.predict(example_measures)
    # print(prediction)
    accuracies.append(accuracy)
    print("Accuracy = ", accuracy )

print("mean = ",sum(accuracies)/len(accuracies))
