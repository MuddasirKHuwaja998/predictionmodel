from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os

app = Flask(__name__)

# Load the trained model and scaler from the current directory
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected.')

    try:
        # Read the CSV file
        df = pd.read_csv(file)

        # Handle NaN values by filling them with the mean of the respective column
        df.fillna(df.mean(), inplace=True)

        # Replace infinite values with the maximum finite value for each column
        for column in df.select_dtypes(include=[np.float64, np.int64]).columns:
            df[column].replace([np.inf, -np.inf], df[column].max(), inplace=True)

        # Feature Engineering: Create interaction features
        df['Interaction_1'] = df['_500dx'] * df['_1000dx']  # Example interaction
        df['Interaction_2'] = df['_2000dx'] * df['_4000dx']  # Another interaction

        # Prepare the data for prediction
        X = df.drop(columns=['Decidera', 'Normoudente', 'Contratto'], errors='ignore')

        # Scale the features
        X_scaled = scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)

        # Prepare results for rendering
        results = pd.DataFrame(predictions, columns=['Decidera', 'Normoudente', 'Contratto'])

        # Create a comparison plot
        plot_file = 'static/prediction_plot.png'
        plt.figure(figsize=(15, 10))

        # Smooth plot function
        def smooth_plot(x, y, label, color):
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)  # Cubic spline
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, label=label, color=color)

        # Check if actual values exist
        if 'Decidera' in df.columns and 'Normoudente' in df.columns and 'Contratto' in df.columns:
            y_actual = df[['Decidera', 'Normoudente', 'Contratto']].values

            for i, column in enumerate(['Decidera', 'Normoudente', 'Contratto']):
                plt.subplot(3, 1, i + 1)
                smooth_plot(np.arange(len(y_actual)), y_actual[:, i], label='Actual', color='blue')
                smooth_plot(np.arange(len(predictions)), predictions[:, i], label='Predicted', color='orange')
                plt.title(f'{column} - Actual vs Predicted')
                plt.xlabel('Sample Index')
                plt.ylabel(column)
                plt.legend()

        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

        return render_template('result.html', predictions=results.values.tolist(), plot=plot_file)

    except Exception as e:
        return render_template('index.html', error=f'Error processing the file: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
