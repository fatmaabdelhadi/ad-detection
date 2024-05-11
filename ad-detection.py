import numpy as np
import pandas as pd
from sklearn import tree
import pydotplus 
from IPython.display import Image
from sklearn.svm import LinearSVC

data = pd.read_csv("ad.csv",low_memory=False)

data.rename(columns={'0': 'height'}, inplace=True)
data.rename(columns={'1': 'width'}, inplace=True)
data.rename(columns={'2': 'aspect_ratio'}, inplace=True)
data.rename(columns={'1558': 'ad?'}, inplace=True)

def toNum(cell):
    if isinstance(cell, (int, float)):
        return cell
    try:
        return float(cell)
    except ValueError:
        return np.nan
    
def seriestoNum(series):
    return series.apply(toNum)

train_data=data.iloc[0:,0:-1].apply(seriestoNum)
train_data=train_data.dropna()

def toLabel(str):
    if str=="ad.":
        return 1
    else:
        return 0
    
train_labels=data.iloc[train_data.index,-1].apply(toLabel)
print(train_labels) # ad or not an ad

print(train_data.tail(20))
print(pd.crosstab([data['width'],data['aspect_ratio']],train_labels))

# Y = train_labels
# X = data.drop(['height','width','aspect_ratio'], axis=1)

X = train_data.drop(['height','width','aspect_ratio'], axis=1)
Y = train_labels

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf = clf.fit(X, Y)

dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['0','1'], filled=True, 
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_png("decision_tree.png")
Image(graph.create_png())