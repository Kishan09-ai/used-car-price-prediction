from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "car_price_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    print("==============================================")
    print("MODEL LOADED SUCCESSFULLY")
    print("==============================================")
    print(f"Model location: {MODEL_PATH}")

except Exception as e:
    model = None

    print("==============================================")
    print("ERROR: MODEL COULD NOT BE LOADED")
    print("==============================================")
    print(f"Expected model location: {MODEL_PATH}")
    print(f"Error: {e}")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PRICE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model is None:
            return """
            <h2>Model Error</h2>
            <p>The trained model could not be loaded.</p>
            <p>Please make sure that
            <b>model/car_price_model.pkl</b> exists.</p>
            """


        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        name = request.form.get("Name")
        location = request.form.get("Location")

        year = float(request.form.get("Year"))
        kilometers = float(
            request.form.get("Kilometers_Driven")
        )

        fuel_type = request.form.get("Fuel_Type")
        transmission = request.form.get("Transmission")
        owner_type = request.form.get("Owner_Type")

        mileage = float(
            request.form.get("Mileage")
        )

        engine = float(
            request.form.get("Engine")
        )

        power = float(
            request.form.get("Power")
        )

        colour = request.form.get("Colour")

        seats = float(
            request.form.get("Seats")
        )

        doors = float(
            request.form.get("No. of Doors")
        )


        # ----------------------------------------------------
        # CALCULATE CAR AGE
        # ----------------------------------------------------

        current_year = 2026

        car_age = current_year - year

        # Prevent negative age if future year is entered
        if car_age < 0:
            car_age = 0


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Name": [name],

            "Location": [location],

            "Year": [year],

            "Kilometers_Driven": [kilometers],

            "Fuel_Type": [fuel_type],

            "Transmission": [transmission],

            "Owner_Type": [owner_type],

            "Mileage": [mileage],

            "Engine": [engine],

            "Power": [power],

            "Colour": [colour],

            "Seats": [seats],

            "No. of Doors": [doors],

            "Car_Age": [car_age]

        })


        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # PREVENT NEGATIVE PRICE
        # ----------------------------------------------------

        prediction = max(0, float(prediction))


        # ----------------------------------------------------
        # CONVERT PRICE INTO INDIAN RUPEES
        #
        # Dataset Price is represented in lakhs.
        #
        # Example:
        # 6.00 = ₹6,00,000
        # ----------------------------------------------------

        price_inr = prediction * 100000


        # ----------------------------------------------------
        # DETERMINE PRICE CATEGORY
        # ----------------------------------------------------

        if prediction < 5:

            category = "Budget"
            category_icon = "💰"

        elif prediction < 10:

            category = "Mid-Range"
            category_icon = "🚘"

        elif prediction < 20:

            category = "Premium"
            category_icon = "⭐"

        else:

            category = "Luxury"
            category_icon = "💎"


        # ----------------------------------------------------
        # RENDER RESULT PAGE
        # ----------------------------------------------------

        return render_template(
            "result.html",

            prediction=prediction,

            price_inr=price_inr,

            category=category,

            category_icon=category_icon,

            car_name=name,

            location=location,

            year=year,

            kilometers=kilometers,

            fuel_type=fuel_type,

            transmission=transmission,

            owner_type=owner_type,

            mileage=mileage,

            engine=engine,

            power=power,

            colour=colour,

            seats=seats,

            doors=doors,

            car_age=car_age
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print("==============================================")
        print("PREDICTION ERROR")
        print("==============================================")
        print(e)

        return f"""
        <div style="
            font-family: Arial;
            padding: 50px;
            text-align: center;
        ">

            <h1>⚠ Prediction Error</h1>

            <p>
                Something went wrong while processing
                your car information.
            </p>

            <p>
                <b>Error:</b> {e}
            </p>

            <br>

            <a href="/">
                ← Go Back
            </a>

        </div>
        """


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print("        AUTOPRICE AI")
    print("   USED CAR PRICE PREDICTION SYSTEM")
    print("==============================================")
    print("")
    print("Developed by:")
    print("Kishan Kr Singh")
    print("Ayush Vardhan")
    print("Rameshwar Teli")
    print("Sheetal Mandal")
    print("")
    print("Department of Artificial Intelligence &")
    print("Machine Learning")
    print("")
    print("Starting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("==============================================")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )