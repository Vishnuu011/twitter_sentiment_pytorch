from flask import Flask, render_template, request
import torch
from src.mlproject.pipelines.predction_pipeline import PredictionPipeline
from src.mlproject.utils import load_object
import os
import sklearn



app = Flask(__name__, template_folder="templates")




@app.route('/', methods=['GET', 'POST'])
def classify():
    sentiment_label = None  
    
    if request.method == 'POST':
        try:
            tweet = request.form.get('tweet', '').strip() 
            if not tweet:
                sentiment_label = "Error: No text provided."
            else:
               
                le = os.path.join("artifact", "le_encoder.pkl")
                leb = load_object(le)

               
                prediction = PredictionPipeline()
                predicted = prediction.predict(text=tweet)
                sentiment_label = leb.inverse_transform(predicted.cpu().numpy())[0]
        except Exception as e:
            sentiment_label = f"Error: {str(e)}"

 
    return render_template('index.html', sentiment=sentiment_label)

if __name__ == '__main__':
    app.run(debug=True)