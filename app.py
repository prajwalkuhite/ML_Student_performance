from flask import Flask , render_template , request ,jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData , PredictPipeline


application = Flask(__name__)

app = application

## Now we are going to route towards home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')   # this home page is basically a first page
    
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results[0])
    
# so by this way we are getting all the predictions


## To test on postman we write this code
@app.route('/postman_data',methods=['GET','POST'])
def predict_datapoint_postman():
    if request.method == 'GET':
        return render_template('home.html')   # this home page is basically a first page
    
    else:
        data = CustomData(
        gender=request.json.get('gender'),
        race_ethnicity=request.json.get('race_ethnicity'),
        parental_level_of_education=request.json.get('parental_level_of_education'),
        lunch=request.json.get('lunch'),
        test_preparation_course=request.json.get('test_preparation_course'),
        reading_score=float(request.json.get('reading_score')),
        writing_score=float(request.json.get('writing_score'))
    )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return jsonify(results=results[0])

if __name__=='__main__':
    app.run(host="0.0.0.0",debug=True)
