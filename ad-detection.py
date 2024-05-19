import numpy as np
import pandas as pd
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns


# Dataset
data = pd.read_csv("ad.csv", low_memory=False)

# Renaming columns
data.rename(columns={'0': 'height'}, inplace=True)
data.rename(columns={'1': 'width'}, inplace=True)
data.rename(columns={'2': 'aspect_ratio'}, inplace=True)
data.rename(columns={'1558': 'ad?'}, inplace=True)

# Removing null values
def toNum(cell):
    if isinstance(cell, (int, float)):
        return cell
    try:
        return float(cell)
    except ValueError:
        return np.nan
    
def seriestoNum(series):
    return series.apply(toNum)

train_data = data.iloc[:, 0:-1].apply(seriestoNum)
train_data = train_data.dropna()

# Converting labels to binary values
def toLabel(str):
    return 1 if str == "ad." else 0

train_labels = data.iloc[train_data.index, -1].apply(toLabel)

print(train_labels)
print(train_data.tail(20))
print(pd.crosstab([data['width'], data['aspect_ratio']], train_labels))

# Preparing data and labels
X = train_data.drop(['height', 'width', 'aspect_ratio'], axis=1)
Y = train_labels

# Splitting data into training and testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
clf_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf_tree.fit(X_train, Y_train)

# Train SVM Classifier
clf_svm = LinearSVC(max_iter=10000)
clf_svm.fit(X_train, Y_train)

# Predictions
Y_pred_tree = clf_tree.predict(X_test)
Y_pred_svm = clf_svm.predict(X_test)

# Evaluation
print("Decision Tree Classifier Report")
print(classification_report(Y_test, Y_pred_tree))
print("Accuracy:", accuracy_score(Y_test, Y_pred_tree))

print("SVM Classifier Report")
print(classification_report(Y_test, Y_pred_svm))
print("Accuracy:", accuracy_score(Y_test, Y_pred_svm))

# Visualize Decision Tree
dot_data = tree.export_graphviz(clf_tree, feature_names=X.columns, class_names=['0', '1'], filled=True, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("decision_tree.png")
Image(graph.create_png())

# Demonstrate Overfitting
clf_tree_overfit = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
clf_tree_overfit.fit(X_train, Y_train)
Y_pred_tree_overfit = clf_tree_overfit.predict(X_test)

print("Overfitted Decision Tree Classifier Report")
print(classification_report(Y_test, Y_pred_tree_overfit))
print("Accuracy:", accuracy_score(Y_test, Y_pred_tree_overfit))

# Model fitting and evaluation
maxdepths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]

trainAcc = np.zeros(len(maxdepths))
testAcc = np.zeros(len(maxdepths))

index = 0
for depth in maxdepths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predTest = clf.predict(X_test)
    trainAcc[index] = accuracy_score(Y_train, Y_predTrain)
    testAcc[index] = accuracy_score(Y_test, Y_predTest)
    index += 1

# Plot of training and testing accuracies
plt.figure(figsize=(10, 6))
plt.plot(maxdepths, trainAcc, 'ro-', maxdepths, testAcc, 'bv--')
plt.legend(['Training Accuracy', 'Test Accuracy'])
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy vs. Max Depth')
plt.grid(True)

# Save plot as an image
plt.savefig("accuracy_plot.png")
plt.show()

# Confusion Matrix
conf = confusion_matrix(Y_test, Y_pred_tree)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True,
             fmt='g', cmap='Blues',
             xticklabels=['Non-Ad', 'Ad'],
             yticklabels=['Non-Ad', 'Ad'])
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.title('Confusion Matrix for Decision Tree Classifier', fontsize=17)
# Save the confusion matrix as an image
plt.savefig("confusion_matrix_decision_tree.png")
plt.show()

# Confusion Matrix for SVM
conf = confusion_matrix(Y_test, Y_pred_svm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True,
             fmt='g', cmap='Blues',
             xticklabels=['Non-Ad', 'Ad'],
             yticklabels=['Non-Ad', 'Ad'])
plt.ylabel('Actual', fontsize=13)
plt.xlabel('Predicted', fontsize=13)
plt.title('Confusion Matrix for SVM Classifier', fontsize=17)
plt.savefig("confusion_matrix_svm.png")
plt.show()