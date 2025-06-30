import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_accuracy = accuracy_score(y_test, tree.predict(X_test))
print("Decision Tree Accuracy:", round(tree_accuracy, 2))

plt.figure(figsize=(14, 7))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=["No", "Yes"])
plt.title("Decision Tree")
plt.show()

simple_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
simple_tree.fit(X_train, y_train)

simple_accuracy = accuracy_score(y_test, simple_tree.predict(X_test))
print("Limited Depth Tree Accuracy:", round(simple_accuracy, 2))

plt.figure(figsize=(12, 6))
plot_tree(simple_tree, filled=True, feature_names=X.columns, class_names=["No", "Yes"])
plt.title("Limited Depth Decision Tree")
plt.show()

forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)


forest_accuracy = accuracy_score(y_test, forest.predict(X_test))
print("Random Forest Accuracy:", round(forest_accuracy, 2))

importances = forest.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=features.index)
plt.title("Features")
plt.xlabel("Score")
plt.show()

scores = cross_val_score(forest, X, y, cv=5)
print("Cv Scores:", scores)
print("CV accuracy:", round(scores.mean(), 2))
