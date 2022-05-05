import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Importing the data-set
df = pd.read_csv("C:\\Users\\Lance Machado\\Desktop\\DSA Projects\\Project Bankruptcy Prevention\\bankruptcy-prevention.csv",
                 delimiter=';')

# Renaming the features
df.rename(columns={' management_risk':'management_risk', ' financial_flexibility':'financial_flexibility',
       ' credibility':'credibility', ' competitiveness':'competitiveness', ' operating_risk':'operating_risk',
                   ' class':'class'},inplace=True)

# Label-encoding
df['class'] = df[['class']].apply(LabelEncoder().fit_transform)

# Droping unnessary features
X = df.drop(['industrial_risk','management_risk','operating_risk','class'],axis=1)
y = df['class']

# Building the model & fitting with training data
log_model = LogisticRegression()
log_model.fit(X.values,y)

# Saving model to disk
pickle.dump(log_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0, 1, 1]]))