from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


model = joblib.load("lr_model.jb")
vectorizer = joblib.load("vectorizer.jb")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    transform = vectorizer.transform([news])
    prediction = model.predict(transform)[0]
    result = " Real " if prediction == 1 else " Fake "
    return render_template('result.html', prediction=result)

@app.route('/dataset_predict', methods=['GET', 'POST'])
def dataset_predict():
    data = pd.read_csv("fake_news.csv") 
    print("Dataset Predict Route Called")
    print("Columns:", data.columns)
    print("Rows Count:", len(data))
    print("CSV columns:", data.columns)

    # Check and clean text/news column
    if 'text' in data.columns:
        data["text"] = data["text"].fillna("").astype(str)
        data["Predicted"] = model.predict(vectorizer.transform(data["text"]))
    elif 'news' in data.columns:
        data["news"] = data["news"].fillna("").astype(str)
        data["Predicted"] = model.predict(vectorizer.transform(data["news"]))
    else:
        return " Error: CSV must have 'text' or 'news' column."

    data.to_csv("predicted_output.csv", index=False)
    return " Predictions saved in predicted_output.csv"


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)

