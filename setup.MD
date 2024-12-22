# Installation Instructions

To set up the project, follow the steps below:

### Step 1: Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/tungducng/diamond-price-prediction.git
cd diamond-price-prediction
```

### Step 2: Set up a Virtual Environment

It's recommended to use a virtual environment to keep dependencies isolated:
For Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

This will install Flask, pandas, scikit-learn, and any other dependencies listed in the requirements.txt.

### Step 4: Run the Application

After the dependencies are installed, you can start the Flask server:

```bash
python app.py
```

The application will start at http://127.0.0.1:5005.

### Step 5: Use the App

1. Open your browser and navigate to `http://127.0.0.1:5005`.

2. **Fill out the form with the required inputs:**

   - The form will require the following diamond attributes:
     - `Carat`
     - `Cut`
     - `Color`
     - `Clarity`
     - `Depth`
     - `Table`
     - `X`
     - `Y`
     - `Z`
   - Enter the appropriate values for each field.

3. **Select the models you want to use for prediction on each row of the Diamonds table:**

   - A table will be displayed with rows of diamond data (carat, cut, color, etc.).
   - For each row, you can select which prediction model (Linear Regression, Decision Tree, or Random Forest) you want to use by checking the corresponding checkbox next to the model name.

4. **Submit the form to view the predicted results:**
   - After filling out the form and selecting your desired models, click the submit button to see the predicted price for the selected diamonds.
   - The predicted results will be shown on the page.

#### Notes:

- If you select multiple models, predictions will be shown for each model.
- Ensure that all necessary input fields are filled out correctly before submitting the form to avoid errors.

### Troubleshooting

1. "ModuleNotFoundError": Ensure that all dependencies are installed correctly by running pip install -r requirements.txt.
2. "FileNotFoundError": Make sure that the file paths for your models (lin_reg.pkl, tree_reg.pkl, forest_reg.pkl, pipeline.pkl, diamonds_test_set_with_index.csv) are correct and the files exist in the project directory.

3. Nếu `scikit-learn` chưa được cài đặt, bạn có thể cài đặt thủ công bằng lệnh:

```bash
pip install scikit-learn
```
