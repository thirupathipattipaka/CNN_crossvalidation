#importing all necessary libraries

from sklearn.metrics import roc_curve,auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

#import data
iris = datasets.load_iris()
X,Y = iris.data, iris.target

Y = MultiLabelBinarizer(Y,classes=[0,1,2])
n_classes = 3

#training and testing data split
X_train, Y_train, X_test, Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

#modeling
clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train,Y_train).decision_function(X_test)


#Compute ROC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i],tpr[i],_ = roc_curve(Y_test[:, i],y_score[:, i])
    roc_auc[i] = auc(fpr[i],tpr[i])

#plot ROC curves
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i],tpr[i],label = 'ROC curve (area = %0.2f' % roc_auc[i])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True positive rate')
    plt.title('ROC example')
    plt.legend(loc="lower right")
    plt.show()



