from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('defect_detection_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        input_data = np.array([float(val) for val in data.values()]).reshape(1, -1)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)[0][0]
        output = "Defect" if prediction > 0.5 else "No Defect"
        return render_template('index.html', prediction=output)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
