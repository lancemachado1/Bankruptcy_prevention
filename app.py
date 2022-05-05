import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app = Flask(__name__)#, template_folder='template')
model = pickle.load(open('C:\\Users\\Lance Machado\\PycharmProjects\\Bankruptcy_prevention\\venv\\model.pkl','rb'))
# 'C:\\Users\\Lance Machado\\PycharmProjects\\Bankruptcy_prevention\\venv\\model.pkl'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 0:
        output = "Company will go bankrupt"
    else:
        output = "Company won't go bankrupt"

    return render_template('index.html', prediction_text= output)


if __name__ == "__main__":
    app.run(debug=True)