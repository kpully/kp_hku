from flask import Flask, jsonify, render_template, request
app = Flask(__name__)

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, train_test_split
import numpy as np
import pandas as pd
from patsy import dmatrices

x=2
df = pd.read_csv('mammographic_masses.data', names = ['BIRADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'])
df = df.replace('?', np.nan)

for column in df:
    df[column] = df[column].astype('float')

df = df.drop('BIRADS', axis=1)
df = df.dropna()

features = ['Age', 'Shape', 'Margin', 'Density']
y, X = dmatrices('Severity ~ C(Shape) + Age + C(Margin) + Density', data=df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression(C=10).fit(X_train, y_train.ravel())


@app.route('/_predict')
def pdct():
    """Add two numbers server side, ridiculous but well..."""
    age = request.args.get('age', 0, type=int)
    shape = request.args.get('shape', 0, type=int)
    margin = request.args.get('margin', 0, type=int)
    density = request.args.get('density', 0, type=int)
    f = np.array([age, shape, margin, density]).reshape(1, -1)
    result = model.predict_proba(f)
    return jsonify(result=features, p_yes=result[0][0], p_no=result[0][1])

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
