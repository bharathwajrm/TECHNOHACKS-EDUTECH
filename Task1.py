import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib


model = joblib.load('trained_model.pkl')

def predict_price():
    try:
        # Get input values from the user
        bedrooms = int(entry_bedrooms.get())
        bathrooms = float(entry_bathrooms.get())
        sqft_living = int(entry_sqft_living.get())
        sqft_lot = int(entry_sqft_lot.get())
        floors = float(entry_floors.get())
        waterfront = int(entry_waterfront.get())
        view = int(entry_view.get())
        condition = int(entry_condition.get())
        grade = int(entry_grade.get())
        sqft_above = int(entry_sqft_above.get())
        sqft_basement = int(entry_sqft_basement.get())
        yr_built = int(entry_yr_built.get())
        yr_renovated = int(entry_yr_renovated.get())
        zipcode = int(entry_zipcode.get())
        lat = float(entry_lat.get())
        long = float(entry_long.get())
        sqft_living15 = int(entry_sqft_living15.get())
        sqft_lot15 = int(entry_sqft_lot15.get())


        new_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                                  condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated,
                                  zipcode, lat, long, sqft_living15, sqft_lot15]],
                                columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                                         'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                                         'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'])


        prediction = model.predict(new_data)
        messagebox.showinfo("Prediction", f"Predicted Price: ${prediction[0]:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

window = tk.Tk()
window.title("House Price Prediction")

input_fields = ['Bedrooms', 'Bathrooms', 'Sqft Living', 'Sqft Lot', 'Floors', 'Waterfront', 'View', 'Condition',
                'Grade', 'Sqft Above', 'Sqft Basement', 'Yr Built', 'Yr Renovated', 'Zipcode', 'Latitude', 'Longitude',
                'Sqft Living15', 'Sqft Lot15']

for i, field in enumerate(input_fields):
    label = tk.Label(window, text=field)
    label.grid(row=i, column=0, padx=10, pady=5)

for i in range(len(input_fields)):
    entry = tk.Entry(window)
    entry.grid(row=i, column=1, padx=10, pady=5)

predict_button = tk.Button(window, text="Predict Price", command=predict_price)
predict_button.grid(row=len(input_fields)+1, columnspan=2, pady=10)

window.mainloop()
