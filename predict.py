import joblib
import pandas as pd

model = joblib.load("data/output/model.pkl")
pipe = joblib.load("data/output/pipe.pkl")

# Input
input_text = "hej regeringen skapa jobb facket #svpol"

word_vector = pipe.transform([input_text])
predictions = model.predict_proba(word_vector)

pred_df = pd.DataFrame({
    "party": model.classes_.tolist(),
    "likeness": predictions[0].tolist()
})

pred_df
