import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)
data = pd.read_csv("mushrooms.csv")
data.head()
print(data)
y = data.iloc[:, 0]  # only first column
y = y.ravel()
X = data.iloc[:, 1:]  # All columns except first column
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=2)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


scores = cross_val_score(knn, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1),scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')

plt.grid(False)
plt.show();
print(scores)

y_pred_knn = knn.predict(X_test)
y_true_knn = y_test
cm = confusion_matrix(y_true_knn, y_pred_knn)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_knn")
plt.ylabel("y_true_knn")
plt.show()

data1 = pd.read_csv("mushrooms.csv")
X = data1.iloc[:, 1:]  # All columns except the last one for features
y = data1.iloc[:, 0]   # The last column for the target


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


rf = RandomForestClassifier(n_estimators=100, random_state=42)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {rf_accuracy * 100:.2f}%")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

scores1 = cross_val_score(rf, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(False)
plt.show();
print(scores1)

y_pred_rf = rf.predict(X_test)
y_true_rf = y_test
cm = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_rf")
plt.ylabel("y_true_rf")
plt.show()

data2 = pd.read_csv("mushrooms.csv")

y = data2.iloc[:, 0]  # only first column
y = y.ravel()
X = data2.iloc[:, 1:]  # All columns except first column


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)


lr_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {lr_accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


scores2 = cross_val_score(log_reg, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(False)
plt.show();
print(scores2)

y_pred_log_reg = log_reg.predict(X_test)
y_true_log_reg = y_test
cm = confusion_matrix(y_true_log_reg, y_pred_log_reg)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_log_reg")
plt.ylabel("y_true_log_reg")
plt.show()

data4 = pd.read_csv("mushrooms.csv")


X = data4.iloc[:, 1:]  # All columns except the last one for features
y = data4.iloc[:, 0]   # The last column for the target
y=y.ravel()


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


X = pd.get_dummies(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


svm = SVC()
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)


svm_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {svm_accuracy * 100:.2f}%")


print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


scores3 = cross_val_score(svm, X, y, cv=5)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='r')
plt.title('Cross-Validation Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.grid(False)
plt.show();
print(scores3)

y_pred_svm = svm.predict(X_test)
y_true_svm = y_test
cm = confusion_matrix(y_true_svm, y_pred_svm)
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred_svm")
plt.ylabel("y_true_svm")
plt.show()