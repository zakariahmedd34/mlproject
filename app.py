from flask import Flask, request,render_template
from src.pipeline.predict_pipline import CustomData,PredictPipeline
from src.logger import logging

app = Flask(__name__,template_folder="templates",static_folder="static")

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/prediction',methods = ['GET','POST'])
def predict_datapoint(): 
    if request.method == "GET":
        return render_template("prediction.html")
    elif request.method == "POST":
        data=CustomData(request.form.get('gender'),request.form.get('ethnicity'),request.form.get('parental_level_of_education'),
        request.form.get('lunch'),
        request.form.get('test_preparation_course'),
       float(request.form.get('reading_score')),
       float(request.form.get('writing_score'))
        )
                
        pred_df = data.get_features_as_datafram()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('prediction.html',results=f'{results[0]:.2f}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
