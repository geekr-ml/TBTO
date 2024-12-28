from flask import Flask, request, render_template, url_for
import pickle
import numpy as np
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the model
model_path = "final_model_for_TBTO.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{model_path}' not found. Ensure the file path is correct.")

# Load training data for SHAP
X_train_path = "X_train_TBTO.xlsx"
try:
    X_train = pd.read_excel(X_train_path)
    feature_names = X_train.columns.tolist()  # Dynamically fetch feature names
except FileNotFoundError:
    raise FileNotFoundError(f"Training data file '{X_train_path}' not found. Ensure the file path is correct.")

# Directory to save SHAP plots
shap_plot_dir = "static/shap_plots"
os.makedirs(shap_plot_dir, exist_ok=True)

# Function to make prediction
def make_prediction(model, features):
    try:
        features = np.array([features])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        return prediction, probabilities
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

# Function to generate SHAP plots and values
def generate_shap_plot_and_values(model, features, X_train, feature_names):
    features_df = pd.DataFrame([features], columns=feature_names)
    try:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(features_df)
        mean_abs_shap_values = np.mean(np.abs(shap_values.values), axis=0)
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Absolute SHAP Value": mean_abs_shap_values
        }).sort_values(by="Mean Absolute SHAP Value", ascending=False)

        shap_table_html = shap_df.to_html(classes="table table-striped", index=False)

        plt.figure(figsize=(8, 6))
        shap.summary_plot(
            shap_values.values,
            features_df,
            feature_names=feature_names,
            show=False
        )
        plot_path = os.path.join(shap_plot_dir, "shap_beeswarm_plot.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        plt.close()

        return plot_path, shap_table_html
    except Exception as e:
        print("Error during SHAP processing:", str(e))
        return None, None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = request.form
        entered_values = {key: form_data.get(key) for key in form_data.keys()}

        # Parse inputs into a list for prediction
        try:
            features = [
                int(form_data.get("rr_tb")),                            # RR-TB Excluding Pre-XDR and XDR-TB
                int(form_data.get("days_in_treatment")),                # No. of Days in Treatment
                float(form_data.get("height")),                         # Height
                float(form_data.get("weight")),                         # Weight
                int(form_data.get("site_of_tb_disease")),               # Site of TB Disease
                int(form_data.get("hiv")),                              # HIV status
                int(form_data.get("rif_resistance_detected")),          # Rif Resistance Detected
                # int(form_data.get("rif_resistance_not_detected")),          # Rif Resistance Detected
                0,
                int(form_data.get("types_of_cases")),                   # Types of Cases
                int(form_data.get("bacteriologically_confirmed")),      # Bacteriologically Confirmed
                float(form_data.get("age"))                             # Age
            ]
        except ValueError:
            return "Error: Invalid input type. Ensure all fields are filled correctly.", 400

        if len(features) != 11:
            return "Error: Incorrect number of features provided. Check the form inputs.", 400

        # Make prediction
        prediction, probabilities = make_prediction(model, features)

        # Map prediction to result
        prediction_map = {
            0: "Cured",
            1: "Died",
            2: "Lost to Follow-up",
            3: "Treatment Complete",
            4: "Treatment Success"
        }
        res = prediction_map.get(prediction, "Unknown Outcome")

        # Calculate confidence score for the prediction (probability of the predicted class)
        confidence_score = probabilities[prediction] * 100

        # Generate a dictionary of all class probabilities
        probability_dict = {
            prediction_map[i]: f"{prob:.2f}%" for i, prob in enumerate(probabilities)
        }


        # Generate SHAP plot and SHAP values table
        shap_plot_path, shap_table_html = generate_shap_plot_and_values(model, features, X_train, feature_names)

        if shap_plot_path is None or shap_table_html is None:
            return "Error generating SHAP values or plot.", 500

        # Render result page
        return render_template(
            "result.html",
            prediction=res,
            confidence_score=f"{confidence_score:.2f}%",  # Confidence score
            probability_dict=probability_dict,
            entered_values=entered_values,
            shap_plot_path=url_for("static", filename="shap_plots/shap_beeswarm_plot.png"),
            shap_table_html=shap_table_html
        )

    except Exception as e:
        print("Error:", str(e))
        return f"An error occurred: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
