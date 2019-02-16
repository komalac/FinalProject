from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib


sds = joblib.load("scaler.model")
clf = joblib.load("H1_status.model")


app = Flask(__name__)


def flatten(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result

@app.route('/')
def index():
    return render_template('Prediction.html')


@app.route('/predict')
def predict():
    
    name=request.args.get("name")
    pwage = float(request.args.get("pwage"))

    if request.args.get("empduration") == "less":
        empdays = 0
    else:
        empdays = 1

    ftp_mapper = {
        "0": [1,0],
        "1": [0,1]
    }
    ftp = request.args.get("ftposition") or "0"
    
    wviolator_mapper = {
        "0": [1,0],
        "1": [0,1]
    }
    wviolator = request.args.get("wvioaltor") or "0"

    dependents_mapper = {
        "no": [1,0],
        "yes": [0,1]        
    }
    dependents = request.args.get("dependents")

    vclass_mapper = {
        "E3Aus": [1,0,0,0],
        "H1B": [0,1,0,0],
        "H1B1C": [0,0,1,0],
        "H1B1S": [0,0,0,1]
    }
    vclass = request.args.get("visaclass")
          
    features = [pwage,
                 empdays,                 
                 ftp_mapper[ftp],                 
                 dependents_mapper[dependents],                 
                 vclass_mapper[vclass],                 
                 wviolator_mapper[wviolator]
                ]


    features = [flatten(features)]
    scaled_features = sds.transform(features)
    spred = clf.predict(scaled_features)
    spred_prob = clf.predict_proba(scaled_features)

    return jsonify([{
                    "case_pred": spred.tolist(), 
                     "case_probs": spred_prob.tolist(),
                     "features": features,
                     "scaled_features": scaled_features.tolist()
                    }])


if __name__ == "__main__":
    app.run(debug=True, port = 5006)
