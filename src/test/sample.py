


import pandas as pd 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import mean_squared_error, confusion_matrix
import seaborn as sns


# Load data
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df_train = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(1)
df_target = df['Survived']

# Train model
lr = LogisticRegression(class_weight={0: 0.9}).fit(df_train.values, df_target)

# Evaluate model
mean_square_error = mean_squared_error(df_target, lr.predict(df_train.values))

print(f"Mean square error: {mean_square_error}")

# Confusion matrix:
sns.heatmap(confusion_matrix(df_target, lr.predict (df_train.values)), annot=True, fmt='d')




path_in = "src/reactive_python_engine.py"
path_out = "src/reactive_python_engine.ts"
with open(path_in, 'r') as file:
    content = file.read().replace("\\n", "\\\\n")
with open(path_out, 'w') as file:
    file.write(f'export const scriptCode = `\n{content}\n`;')