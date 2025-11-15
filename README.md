House-Price-Prediction
This project focuses on building a highly accurate model for house price prediction using a deep learning approach implemented with TensorFlow. The goal is to estimate the final sale price of houses in King County, Washington (Seattle area), based on a comprehensive set of features.

House Price Prediction using Artificial Neural Networks (ANN)
Overview
This project implements an Artificial Neural Network (ANN) using TensorFlow/Keras to predict house sale prices based on various features. The goal is to build a robust regression model capable of estimating a house's price given its attributes.

Dataset
The project utilizes the King County House Sales dataset (read from /content/kc_house_data.csv).

Dataset Details
Total Entries: 21,613 records.

Total Features: 21 columns.

Target Variable: price.

Key Features include: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, grade, yr_built, zipcode, lat, and long.

Prerequisites (Dependencies)
To run this notebook, you will need the following Python libraries:

numpy

pandas

tensorflow

scikit-learn (for scaling and train/test split)

The libraries are imported in the notebook.

Methodology and Steps
The following steps outline the data processing and model development within the notebook:

1. Data Loading and Initial Analysis
Load Data: The kc_house_data.csv file is loaded into a Pandas DataFrame.

Data Structure: Initial checks are performed using df.head(), df.shape, and df.info() to inspect the data, identify the 21 columns, and confirm no missing values are present.

2. Data Preprocessing and Feature Engineering
Target Reordering: The target column, price, is moved to the end of the DataFrame.

Feature Removal: The non-predictive features id and date are dropped from the dataset.

Feature/Target Separation: The remaining features are assigned to X, and the target variable (price) is assigned to y.

Train-Test Split: The data is split into training and testing sets (X_train, X_test, y_train, y_test), reserving approximately 25% of the data for testing.

Feature Scaling (Standardization): The numerical features (X data) and the target variable (y data) are normalized using StandardScaler to ensure all values are on a comparable scale, which is crucial for Neural Network training.

3. Model Building and Training
Model Architecture: An Artificial Neural Network (ANN) is built using a Sequential model from TensorFlow/Keras.

Input Layer: Dense layer with 100 units and relu activation.

Hidden Layer: Dense layer with 50 units and relu activation.

Output Layer: Dense layer with 1 unit (linear activation is implicit for regression).

Model Compilation: The model is compiled using the Adam optimizer.

Loss Function: Mean Squared Error (loss='mse').

Metric: Mean Absolute Error (metrics=['mae']).

Training: The model is trained for 100 epochs with a batch size of 128. A validation split of 0.2 is used to monitor performance on unseen data during training.

4. Evaluation and Prediction
Prediction: The trained model is used to generate predictions (y_pred) on the scaled test set (X_test).

Inverse Transformation: The scaled predictions (y_pred) and the scaled true values (y_test) are converted back to their original dollar scale using the inverse transformation of the scaler object.

Output: The actual and predicted prices for the first few samples of the test set are displayed, showing the model's accuracy on the final scale.
