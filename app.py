from flask import Flask, redirect, url_for, render_template, request, send_file, Response,jsonify
import pandas as pd
import numpy as np
import pickle as pkl
import torch
from sklearn import preprocessing
from pytorch_tabnet.tab_model import TabNetClassifier
from kustom import TabNetWithDropout
from process import preparation, generate_response
# label_encoder object knows how to understand word labels.
le = preprocessing.LabelEncoder()

preparation()


with open("model/tabnet_model123.pkl" ,"rb") as file:
    loaded_tabnet_model = pkl.load(file)

CKDORNOT = ['Sehat',  'Sakit Ginjal']

app = Flask(__name__)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/single2')
def single2():
    return render_template('single2.html')

@app.route('/single3')
def single3():
    return render_template('single3.html')

@app.route('/single4')
def single4():
    return render_template('single4.html')

@app.route('/trips', methods=['GET', 'POST'])
def trips():
    if request.method == 'POST':
        input_data = {
            'age': request.form['age'],
            'blood_pressure': request.form['blood_pressure'],
            'urine_specific_gravity': request.form['urine_specific_gravity'],
            'albumin': request.form['albumin'],
            'sugar': request.form['sugar'],
            'red_blood_cells': request.form['red_blood_cells'],
            'pus_cell': request.form['pus_cell'],
            'pus_cell_clumps': request.form['pus_cell_clumps'],
            'bacteria': request.form['bacteria'],
            'blood_glucose_random': request.form['blood_glucose_random'],
            'blood_urea': request.form['blood_urea'],
            'serum_creatinine': request.form['serum_creatinine'],
            'sodium': request.form['sodium'],
            'potassium': request.form['potassium'],
            'haemoglobin': request.form['haemoglobin'],
            'packed_cell_volume': request.form['packed_cell_volume'],
            'white_blood_cell_count': request.form['white_blood_cell_count'],
            'red_blood_cell_count': request.form['red_blood_cell_count'],
            'hypertension': request.form['hypertension'],
            'diabetes_mellitus': request.form['diabetes_mellitus'],
            'coronary_artery_disease': request.form['coronary_artery_disease'],
            'appetite': request.form['appetite'],
            'peda_edema': request.form['peda_edema'],
            'anemia': request.form['anemia']
        }


        dt = pd.DataFrame([input_data])
        dt['red_blood_cells'] = le.fit_transform(dt['red_blood_cells'])
        dt['pus_cell'] = le.fit_transform(dt['pus_cell'])
        dt['pus_cell_clumps'] = le.fit_transform(dt['pus_cell_clumps'])
        dt['bacteria'] = le.fit_transform(dt['bacteria'])
        dt['hypertension'] = le.fit_transform(dt['hypertension'])
        dt['diabetes_mellitus'] = le.fit_transform(dt['diabetes_mellitus'])
        dt['coronary_artery_disease'] = le.fit_transform(dt['coronary_artery_disease'])
        dt['appetite'] = le.fit_transform(dt['appetite'])
        dt['peda_edema'] = le.fit_transform(dt['peda_edema'])
        dt['anemia'] = le.fit_transform(dt['anemia'])
        
        new_datatensor = dt.apply(pd.to_numeric, errors='coerce')
        new_datatensor = new_datatensor.values
        new_datatensor = new_datatensor.astype(np.float32)
        new_datatensor = torch.tensor(new_datatensor)
        new_datatensor = new_datatensor.to(loaded_tabnet_model.device)
        prediction_value = loaded_tabnet_model.predict(new_datatensor)
        prediction_value = CKDORNOT[prediction_value[0]]
        yes = 'Hasil prediksi menunjukkan indikasi adanya masalah pada ginjal Anda. Kami sarankan Anda untuk berkonsultasi dengan dokter spesialis ginjal untuk pemeriksaan lebih lanjut dan mendapatkan penanganan yang tepat.'
        no = 'Berdasarkan analisis kami, saat ini tidak ada indikasi masalah ginjal pada Anda. Ini adalah kabar yang baik!'
        if prediction_value == "Sehat":
            return render_template('trips.html', result=prediction_value, input_data=input_data, out = no)
        else:
            return render_template('trips.html', result=prediction_value, input_data=input_data, out = yes)
    return render_template('trips.html')

@app.route("/get", methods=["GET", "POST"])
def get_bot_response():
    if request.method == "POST":
        user_input = request.form["msg"]
        result = generate_response(user_input)
        print("Methodnya post bree")
        return jsonify (result)
    else:
        print("Methodnya get bree")
        return 0
        
@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    response = generate_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == '__main__':
    app.run(debug=True)