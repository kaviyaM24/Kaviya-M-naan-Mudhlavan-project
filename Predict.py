importimport pandas as pd
import joblib

# load model and feature columns
model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

# load new data
new_data = pd.read_csv("new_data.csv")  # must match format

# preprocess
new_data = pd.get_dummies(new_data)
new_data = new_data.reindex(columns=columns, fill_value=0)

# predict
predictions = model.predict(new_data)
print("predicted prices:", predictions)
