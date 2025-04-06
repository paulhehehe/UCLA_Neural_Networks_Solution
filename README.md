# UCLA_Admission_Predictor
This application predicts UCLA graduate program admission chances using a Neural Network model, featuring an interactive Streamlit interface for real-time probability assessment.

## Features
Comprehensive data preprocessing pipeline with logging and error handling
Interactive form for academic profile input
Pre-trained Neural Network model with MinMax scaling
Real-time prediction with visual feedback (success/error messages)
Model training visualization via loss curve

## Data Processing
The project includes robust data loading and preprocessing with:

Error handling for missing files and data corruption
Logging of all data transformations
Data validation checks

## Feature Engineering
Feature engineering includes:

Feature scaling: MinMax normalization via pre-fitted scaler
Input structuring: Maintains exact feature order required by model
Dimensionality: Expands 6 raw inputs to 12 model features through encoding

## Technologies Used
Python: Core application logic
Streamlit: Interactive web interface
Scikit-learn: Neural Network classifier
Pickle: Model and scaler serialization
Matplotlib/Seaborn: Loss curve visualization

## Model
Algorithm: Neural Network Classifier

Input Features:
GRE Score
TOEFL Score
University Rating (one-hot encoded)
SOP Strength
LOR Strength
CGPA
Research Experience

Output: Binary prediction (1 = Admitted, 0 = Not Admitted)

Training Insight: Loss curve visualization (PNG)

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/paulhehehe/UCLA_Neural_Networks_Solution.git
   cd UCLA_Neural_Networks_Solution

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

#### Thank you for using the UCLA_Admission_Predictor! Feel free to share your feedback.
