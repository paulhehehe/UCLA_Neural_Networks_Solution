# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Swapnil Kangralkar',
#     license='',
# )

from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_loss_curve
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_Neuralmodel
from src.models.predict_model import evaluate_model
import numpy as np

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/Admission.csv"
    df = load_and_preprocess_data(data_path)

    # Create dummy variables and separate features and target
    clean_df = create_dummy_vars(df)

    # Train the logistic regression model
    model, x_test_scaled, y_test = train_Neuralmodel(clean_df)

    # Evaluate the model
    plot_loss_curve(model)
    accuracy, confusion_mat = evaluate_model(model, x_test_scaled, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{confusion_mat}")

