import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset loaded successfully!")

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns)

# Display data types
print("\nData types:")
print(df.dtypes)

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Number of unique crops
print("\nNumber of unique crops:")
print(df["label"].nunique())

# List of unique crops
print("\nCrop names:")
print(df["label"].unique())

# Number of samples for each crop
print("\nSamples per crop:")
print(df["label"].value_counts())

# Crop distribution
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df,
    x="label",
    order=df["label"].value_counts().index
)
plt.title("Crop Distribution")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of numerical features
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]
df[features].hist(
    figsize=(14, 10),
    bins=20,
    edgecolor="black"
)
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# Correlation matrix
correlation = df[features].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Boxplots for detecting outliers
plt.figure(figsize=(14, 8))
sns.boxplot(data=df[features])
plt.title("Boxplots of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# Preparing data for Machine Learning
# ============================================================

# Input features
X = df[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

# Target variable
y = df["label"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nShape of X:")
print(X.shape)

print("\nShape of y:")
print(y.shape)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("\nOriginal crop names:")
print(y.head())
print("\nEncoded target values:")
print(y_encoded[:5])
print("\nCrop-to-number mapping:")
for crop, number in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(crop, "->", number)

# Split the dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target shape:")
print(y_train.shape)

print("\nTesting target shape:")
print(y_test.shape)

# Feature scaling
scaler = StandardScaler()
# Fit scaler only on training data
X_train_scaled = scaler.fit_transform(X_train)
# Use the same scaler to transform testing data
X_test_scaled = scaler.transform(X_test)
print("\nScaled training data shape:")
print(X_train_scaled.shape)
print("\nScaled testing data shape:")
print(X_test_scaled.shape)

# ============================================================
# Logistic Regression
# ============================================================

logistic_model = LogisticRegression(max_iter=1000)
# Train the model
logistic_model.fit(X_train_scaled, y_train)
print("\nLogistic Regression model trained successfully!")
# Make predictions on test data
y_pred_logistic = logistic_model.predict(X_test_scaled)
# Calculate accuracy
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)
print("\nLogistic Regression Accuracy:")
print(logistic_accuracy)

# ============================================================
# K-Nearest Neighbors (KNN)
# ============================================================

knn_model = KNeighborsClassifier(n_neighbors=5)
# Train the model
knn_model.fit(X_train_scaled, y_train)
print("\nKNN model trained successfully!")
# Make predictions
y_pred_knn = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\nKNN Accuracy:")
print(knn_accuracy)

# ============================================================
# Decision Tree
# ============================================================

decision_tree_model = DecisionTreeClassifier(
    random_state=42
)
# Train the model
decision_tree_model.fit(X_train, y_train)
print("\nDecision Tree model trained successfully!")
# Make predictions
y_pred_tree = decision_tree_model.predict(X_test)
# Calculate accuracy
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print("\nDecision Tree Accuracy:")
print(tree_accuracy)

# ============================================================
# Random Forest
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Train the model
random_forest_model.fit(X_train, y_train)
print("\nRandom Forest model trained successfully!")
# Make predictions
y_pred_rf = random_forest_model.predict(X_test)
# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:")
print(rf_accuracy)

# ============================================================
# Support Vector Machine (SVM)
# ============================================================

svm_model = SVC(
    kernel="rbf",
    random_state=42
)
# Train the model
svm_model.fit(X_train_scaled, y_train)
print("\nSVM model trained successfully!")
# Make predictions
y_pred_svm = svm_model.predict(X_test_scaled)
# Calculate accuracy
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("\nSVM Accuracy:")
print(svm_accuracy)

# ============================================================
# Naive Bayes
# ============================================================

naive_bayes_model = GaussianNB()
# Train the model
naive_bayes_model.fit(X_train, y_train)
print("\nNaive Bayes model trained successfully!")
# Make predictions
y_pred_nb = naive_bayes_model.predict(X_test)
# Calculate accuracy
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print("\nNaive Bayes Accuracy:")
print(nb_accuracy)

# ============================================================
# Model Comparison
# ============================================================

model_results = {
    "Logistic Regression": logistic_accuracy,
    "KNN": knn_accuracy,
    "Decision Tree": tree_accuracy,
    "Random Forest": rf_accuracy,
    "SVM": svm_accuracy,
    "Naive Bayes": nb_accuracy
}

print("\n========================================")
print("MODEL ACCURACY COMPARISON")
print("========================================")

for model, accuracy in model_results.items():
    print(f"{model}: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# ============================================================
# Classification Reports
# ============================================================

print("\n========================================")
print("RANDOM FOREST CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=label_encoder.classes_
    )
)

print("\n========================================")
print("NAIVE BAYES CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_test,
        y_pred_nb,
        target_names=label_encoder.classes_
    )
)

joblib.dump(random_forest_model, "models/random_forest_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("\nRandom Forest model saved successfully!")
print("Label encoder saved successfully!")