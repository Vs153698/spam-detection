import pickle
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('newVectorizer.pkl', 'rb') as f:
    newtfidf = pickle.load(f)

with open('newECModel.pkl', 'rb') as f:
    newModel = pickle.load(f)


app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    data = request.get_json()
    text = data['text']
    processed_text = tfidf.transform([text])
    extraProcesses = newtfidf.transform([text])
    result1 = newModel.predict(extraProcesses)
    result = model.predict(processed_text)
    flag = ""
    if (result.tolist()[0] == 0 and result1.tolist()[0] == 0):
        flag = "Not Spam"
    else:
        flag = "Spam"
    return {'result': flag}

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json()
    text = data['text']
    processed_text = tfidf.transform([text])
    extraProcesses = newtfidf.transform([text])
    result1 = newModel.predict(extraProcesses)
    result = model.predict(processed_text)
    print(result1.tolist(),result.tolist())
    flag = ""
    if (result.tolist()[0] == 0 and result1.tolist()[0] == 0):
        flag = "Not Spam"
    else:
        flag = "Spam"
    return {'result': flag}


if __name__ =="__main__":
    app.run()


