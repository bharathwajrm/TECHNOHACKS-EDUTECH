import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

model, vectorizer = joblib.load('sentiment_analysis_model.pkl')
def predict_sentiment():
    new_post = entry_post.get()
    if new_post.strip() == '':
        messagebox.showwarning("Warning", "Please enter a post.")
        return
    new_post_features = vectorizer.transform([new_post])
    predicted_sentiment = model.predict(new_post_features)
    messagebox.showinfo("Prediction", f"Predicted Sentiment: {predicted_sentiment[0]}")


window = tk.Tk()
window.title("Social Media Sentiment Analysis")


label = tk.Label(window, text="Enter a Social Media Post:")
label.pack(pady=5)
entry_post = tk.Entry(window, width=50)
entry_post.pack(pady=5)

predict_button = tk.Button(window, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack(pady=5)


window.mainloop()
