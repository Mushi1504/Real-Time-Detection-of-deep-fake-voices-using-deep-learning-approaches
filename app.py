# Importing essential libraries and modules

from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import requests
import config
import pickle
import io
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_dic= ['fake','real']



from model_predict  import voice_detector

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Deep Fake Audio Detection'
    return render_template('index.html', title=title)

# render crop recommendation form page

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Deep Fake Audio Detection'

    if request.method == 'POST':
        #if 'file' not in request.files:
         #   return redirect(request.url)

            file = request.files.get('file')

           # if not file:
            #    return render_template('disease.html', title=title)

            #img = Image.open(file)
            file.save('output.wav')


            prediction =voice_detector("output.wav")

            #prediction = (str(disease_dic[prediction]))

            #prediction="5"

            print("print the blood group of the candidate ",prediction)

            if prediction=="Fake":
                    class_rust="FAKE"


            elif prediction=="Real":
                    class_rust="REAL"





            return render_template('rust-result.html', prediction=prediction,precaution=class_rust,title=title)
        #except:
         #   pass
    return render_template('rust.html', title=title)


# render disease prediction result page


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
