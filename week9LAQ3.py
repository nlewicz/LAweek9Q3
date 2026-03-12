from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

red_file = "winequality-red.csv"
white_file = "winequality-white.csv"

red_df = pd.read_csv(red_file, sep=";")
white_df = pd.read_csv(white_file, sep=";")

wine_df = pd.concat([red_df, white_df], ignore_index=True)

wine_df["quality_binary"] = (wine_df["quality"] > 6).astype(int)

X = wine_df.drop(columns=["quality", "quality_binary"])
y = wine_df["quality_binary"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

best_depth = None
best_dt_f1 = 0

for depth in [3, 5, 7, 9, 11]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)
    score = f1_score(y_val, y_val_pred)
    if score > best_dt_f1:
        best_dt_f1 = score
        best_depth = depth

best_k = None
best_knn_f1 = 0

for k in [3, 5, 7, 9, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_val_pred = model.predict(X_val_scaled)
    score = f1_score(y_val, y_val_pred)
    if score > best_knn_f1:
        best_knn_f1 = score
        best_k = k

dt_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_model.fit(X_train_scaled, y_train)

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

models = {
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model
}

print("Best Decision Tree max_depth:", best_depth)
print("Best KNN n_neighbors:", best_k)

print("\nValidation Results")
print("-" * 40)
for name, model in models.items():
    y_val_pred = model.predict(X_val_scaled)
    print(name)
    print("Accuracy :", accuracy_score(y_val, y_val_pred))
    print("Precision:", precision_score(y_val, y_val_pred))
    print("Recall   :", recall_score(y_val, y_val_pred))
    print("F1 Score :", f1_score(y_val, y_val_pred))
    print("-" * 40)

print("\nTest Results")
print("-" * 40)
for name, model in models.items():
    y_test_pred = model.predict(X_test_scaled)
    print(name)
    print("Accuracy :", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall   :", recall_score(y_test, y_test_pred))
    print("F1 Score :", f1_score(y_test, y_test_pred))
    print("-" * 40)