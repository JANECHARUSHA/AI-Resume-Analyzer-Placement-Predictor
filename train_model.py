import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

X = data.drop("Placement", axis=1)
y = data["Placement"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

# Save model
pickle.dump(model,open("model.pkl","wb"))

print("Model trained and saved!")