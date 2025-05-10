# ml-project-pune-house-prices
This project uses machine learning techniques to predict property prices in Pune based on features like location, square footage, number of bathrooms, and BHK. The model is built using Linear Regression and includes data preprocessing, feature engineering, and model evaluation.

📝 Project Overview
The goal of this project is to predict the price of properties in Pune using data about various property features. The project uses Linear Regression to build the prediction model and applies various techniques like outlier removal, data cleaning, and exploratory data analysis (EDA) to improve model performance.

🚀 Key Features
Predicts Property Prices: Uses features like location, BHK, total_sqft, and price to predict property prices.
Cross-Validation: Uses ShuffleSplit and cross-validation to evaluate the model's accuracy, achieving around 85% accuracy on average.
Outlier Removal: Implements methods to remove outliers from the data based on price per square foot (price_per_sqft).
Data Preprocessing: Performs necessary data transformations like feature scaling, handling missing values, and encoding categorical variables.
Exploratory Data Analysis (EDA): Analyzes the data visually to discover trends, correlations, and key insights.

📊 Tools & Technologies
Python: Programming language used for data processing and machine learning.
Pandas: For data manipulation and preprocessing.
NumPy: For numerical operations and calculations.
Matplotlib & Seaborn: For data visualization.
Scikit-learn: For machine learning algorithms, model evaluation, and validation techniques.
Jupyter Notebook: For running and documenting the code in an interactive environment.

🛠️ Project Setup
1. Clone the repository:
git clone https://github.com/your-username/PropertyPricePrediction.git
cd PropertyPricePrediction

2. Install the required libraries:
pip install -r requirements.txt

3. Run the Jupyter notebook:
jupyter notebook Pune_House_Price_Prediction.ipynb

🔍 Model Evaluation
The model's performance is evaluated using cross-validation. Below are the accuracy scores from the model:
[0.87729294, 0.87158074, 0.82825079, 0.89763339, 0.81068616]

📝 Code Files
Pune_House_Price_Prediction.ipynb: Jupyter notebook containing the full workflow, from data cleaning to model evaluation.
model.pkl: Saved model file after training, ready for deployment.
requirements.txt: List of Python packages required for the project.
README.md: Project overview and setup instructions.

📂 Directory Structure
PropertyPricePrediction/
├── Pune_House_Price_Prediction.ipynb       # Jupyter notebook with full workflow
├── model.pkl                               # Saved ML model
├── requirements.txt                        # List of required Python packages
└── README.md                               # Project overview

