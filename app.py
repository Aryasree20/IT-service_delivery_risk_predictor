from flask import Flask, render_template, request
import numpy as np
import os
from src.risk_predictor.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "✅ Training Successful!"

@app.route('/predict', methods=['POST','GET'])
def index():
    
    if request.method == 'POST':
        try:
            # Collect inputs from form
            planned_duration_days = float(request.form['planned_duration_days'])
            actual_duration_days = float(request.form['actual_duration_days'])
            team_size = int(request.form['team_size'])
            num_bugs = int(request.form['num_bugs'])
            num_change_requests = int(request.form['num_change_requests'])
            budget_overrun_pct = float(request.form['budget_overrun_pct'])

            # Prepare input
            data = np.array([
                planned_duration_days,
                actual_duration_days,
                team_size,
                num_bugs,
                num_change_requests,
                budget_overrun_pct
            ]).reshape(1, -1)

            # Predict
            obj = PredictionPipeline()
            prediction = obj.predict(data)

            return render_template('results.html', prediction=prediction)

        except Exception as e:
            print("❌ Exception: ", e)
            return "Something went wrong"
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    
