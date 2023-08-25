# Fruit Image Classification using Jupyter Notebook

This project involves building a fruit image classification model using TensorFlow and Keras in a Jupyter Notebook environment. The goal is to train a machine learning model to accurately classify images of different fruits.

## Project Overview

In this project, we use a dataset containing images of various fruits, each belonging to a specific category. We will build a neural network model to predict the fruit category based on input images. The project includes the following steps:

1. **Data Preparation**: Loading and preprocessing the dataset using TensorFlow's `ImageDataGenerator` to perform data augmentation and normalization.

2. **Model Building**: Constructing a neural network model using a pre-trained base model, adding additional layers for classification, and configuring the model's architecture.

3. **Model Training**: Training the model using the prepared dataset, monitoring its performance using validation data, and adjusting hyperparameters.

4. **Evaluation and Visualization**: Evaluating the model's performance on validation data and visualizing training and validation loss trends.

5. **Inference**: Making predictions using the trained model on new, unseen images.

## How to Use

1. **Clone the Repository**: Clone this repository to your local machine.

2. **Install Dependencies**: Ensure you have the necessary dependencies installed. You can install them using the following command:

    ```bash
    pip install tensorflow matplotlib
    ```

3. **Download the Dataset**: Download the fruit image dataset and place it in a directory named `data`.

4. **Open the Jupyter Notebook**: Open the Jupyter Notebook (`fruit_classification.ipynb`) using Jupyter Notebook or JupyterLab.

5. **Run the Notebook**: Follow the step-by-step instructions in the notebook to preprocess the data, build the model, train it, and evaluate its performance.

6. **Explore Results**: Examine the training and validation loss trends using plots generated in the notebook. Make predictions using the trained model on new images.

## File Structure

- `data/`: Directory containing the fruit image dataset.
- `fruit_classification.ipynb`: Jupyter Notebook with the project implementation.
- `README.md`: This documentation file.

## Acknowledgments

This project was completed as part of a machine learning course.
