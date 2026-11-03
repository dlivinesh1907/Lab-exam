
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset
iris = load_iris()

# Input and output data
X = iris.data
y = iris.target

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Save trained model
joblib.dump(model, "iris_model.pkl")

print("Model trained successfully!")
print("Model saved as iris_model.pkl")
