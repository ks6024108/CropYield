from flask import Flask, request, render_template
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io, base64

# Loading models
model = joblib.load('DTRJoblib.pkl')
preprocessor = joblib.load('PreJoblib.pkl')

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item = request.form['Item']

        # Prepare and transform input
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = model.predict(transformed_features)[0]

        # Generate temperature vs yield plot
        temps = list(range(10, 45))
        yields = []
        for t in temps:
            temp_features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, t, Area, Item]], dtype=object)
            transformed_temp_features = preprocessor.transform(temp_features)
            yield_pred = model.predict(transformed_temp_features)[0]
            yields.append(yield_pred)

        # Plot the graph
        fig, ax = plt.subplots()
        ax.plot(temps, yields, color='green', marker='o')
        ax.set_xlabel('Average Temperature')
        ax.set_ylabel('Predicted Yield')
        ax.set_title('Effect of Temperature on Crop Yield')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return render_template('result.html', prediction=round(prediction, 2), plot_url=graph)

if __name__ == "__main__":
    app.run(debug=True)
