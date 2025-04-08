# UCLA_Admission_Predictor
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://uclaneuralnetwork.streamlit.app/)

password - streamlit

This application predicts UCLA graduate program admission chances using a Neural Network model, featuring an interactive Streamlit interface for real-time probability assessment.

## Features
- Comprehensive data preprocessing pipeline with logging and error handling.
- Interactive form for academic profile input.
- Pre-trained Neural Network model with MinMax scaling.
- Real-time prediction with visual feedback (success/error messages).
- Model training visualization via loss curve.
- Accessible via Streamlit Community Cloud.

## Dataset
The application is trained on admission data for graduate programs, including features like:
- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP) Strength
- Letter of Recommendation (LOR) Strength
- Cumulative Grade Point Average (CGPA)
- Research Experience

## Technologies Used
- **Streamlit**: Interactive web interface.
- **Scikit-learn**: Neural Network classifier.
- **Pandas** and **NumPy**: Data preprocessing and manipulation.
- **Pickle**: Model and scaler serialization.
- **Matplotlib/Seaborn**: Visualization of the loss curve.

## Model
The predictive model used is a Neural Network classifier. It applies preprocessing steps such as MinMax normalization via a pre-fitted scaler and expands raw input features through encoding. The final output is a binary prediction indicating admission status.

## Future Enhancements
* Adding support for multiple universities.
* Incorporating more interactive visualizations.
* Implementing feature importance analysis for transparency.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/paulhehehe/UCLA_Neural_Networks_Solution.git
   cd UCLA_Neural_Networks_Solution
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

#### Thank you for using the UCLA Admission Predictor! Feel free to share your feedback.

