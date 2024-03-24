import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib


model = joblib.load('employee_turnover_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')


def predict_turnover():

    age = int(entry_age.get())
    business_travel = entry_business_travel.get()
    daily_rate = int(entry_daily_rate.get())

    input_data = pd.DataFrame({
        'Age': [age],
        'DailyRate': [daily_rate],

    })

    if business_travel not in label_encoders['BusinessTravel'].classes_:
        messagebox.showwarning("Warning", f"Invalid value for BusinessTravel: {business_travel}")
        return

    business_travel_encoded = label_encoders['BusinessTravel'].transform([business_travel])[0]
    input_data['BusinessTravel'] = business_travel_encoded

    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "Likely to leave"
    else:
        result = "Not likely to leave"

    # Show prediction result in a message box
    messagebox.showinfo("Prediction Result", f"The employee is {result}.")


# Create tkinter window
window = tk.Tk()
window.title("Employee Turnover Prediction")

label_age = tk.Label(window, text="Age:")
label_age.grid(row=0, column=0, padx=10, pady=5)
entry_age = tk.Entry(window)
entry_age.grid(row=0, column=1, padx=10, pady=5)

label_business_travel = tk.Label(window, text="Business Travel:")
label_business_travel.grid(row=1, column=0, padx=10, pady=5)
entry_business_travel = tk.Entry(window)
entry_business_travel.grid(row=1, column=1, padx=10, pady=5)

label_daily_rate = tk.Label(window, text="Daily Rate:")
label_daily_rate.grid(row=2, column=0, padx=10, pady=5)
entry_daily_rate = tk.Entry(window)
entry_daily_rate.grid(row=2, column=1, padx=10, pady=5)

predict_button = tk.Button(window, text="Predict Turnover", command=predict_turnover)
predict_button.grid(row=3, columnspan=2, pady=10)


window.mainloop()
