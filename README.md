
# Fraud Detection Model with XGBoost and AWS SageMaker

This project demonstrates how to build a fraud detection model using XGBoost and deploy it on AWS SageMaker. The model is trained using a dataset containing transaction details, and the goal is to predict fraudulent transactions.

## Overview

1. **Data Preprocessing & Feature Engineering**
   - Load the transaction data from an S3 bucket.
   - Perform feature engineering, including transforming date columns, creating new features like age, transaction hour, day of week, and distance to the merchant, and one-hot encoding categorical features.
   - Apply SMOTE to handle class imbalance in the dataset.
   - Scale numerical features using `StandardScaler`.

2. **Model Training with XGBoost**
   - Define and train an XGBoost classifier using AWS SageMaker’s built-in XGBoost container.
   - Hyperparameters are tuned based on model requirements.
   - The model is trained using both training and validation datasets, which are uploaded to S3.

3. **Model Deployment on AWS SageMaker**
   - The trained model is deployed on an AWS SageMaker endpoint.
   - The endpoint is used for real-time prediction of fraud using sample transaction data.

4. **Testing the Endpoint**
   - Send a test transaction to the deployed endpoint and get the fraud prediction.

## Prerequisites

- **AWS Account**: Ensure that you have access to AWS services such as SageMaker and S3.
- **IAM Role**: Ensure the AWS IAM role used has sufficient permissions to access S3, SageMaker, and other necessary services.
- **Libraries**: The following libraries are required to run the project:
    - `xgboost`
    - `imblearn`
    - `boto3`
    - `sagemaker`
    - `pandas`
    - `numpy`
    - `sklearn`
    - `joblib`

## Installation

First, install the necessary libraries by running the following:

```bash
pip install xgboost imblearn boto3 sagemaker pandas numpy scikit-learn joblib
```

## Setup

1. **S3 Bucket**: Store your dataset (fraudTrain.csv) in an S3 bucket, and update the `bucket` and `object_key` variables in the script accordingly.
2. **IAM Role**: Ensure your IAM role has the necessary permissions to interact with SageMaker and S3. The role should have `sagemaker.amazonaws.com` policies attached.
3. **Run the script**: Execute the provided Python script either in a SageMaker notebook instance or locally.

## How to Run

1. **Download the dataset**: The dataset is loaded directly from an S3 bucket using the `boto3` library.
2. **Feature Engineering**: The script performs various preprocessing tasks including converting dates, handling categorical variables, and scaling numerical features.
3. **Model Training**: XGBoost is trained on the processed dataset and the model is saved in the S3 bucket.
4. **Model Deployment**: Once trained, the model is deployed to a SageMaker endpoint for real-time inference.
5. **Prediction**: After deployment, use the endpoint to predict whether a transaction is fraudulent.

### Hyperparameters

The following hyperparameters are used in training the XGBoost model:

- **objective**: `binary:logistic` (binary classification task)
- **eval_metric**: `logloss` (logarithmic loss function)
- **num_round**: 300 (number of boosting rounds)
- **eta**: 0.1 (learning rate)
- **max_depth**: 5 (maximum depth of trees)
- **subsample**: 0.8 (subsampling ratio)
- **colsample_bytree**: 0.8 (subsample ratio of features per tree)
- **reg_alpha**: 0.2 (L1 regularization term)
- **reg_lambda**: 0.2 (L2 regularization term)
- **scale_pos_weight**: computed from the class imbalance ratio

### Model Evaluation

The model is evaluated using the following metrics:

- **ROC-AUC score**
- **F1-score**
- **Precision**
- **Recall**

### Sample Prediction

After deployment, sample data can be sent to the SageMaker endpoint, and the model will return a prediction score, which is then transformed into a probability using the logistic sigmoid function.

```python
score = response['predictions'][0].get('score')
probability = 1 / (1 + np.exp(-score))
print(f"Probability of positive class (fraud): {probability}")
```

## Conclusion

This project demonstrates how to preprocess transactional data, train an XGBoost model, and deploy it on AWS SageMaker for fraud detection tasks. The model can be used for real-time predictions once deployed, helping businesses identify potentially fraudulent transactions in real time.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- AWS for providing the infrastructure to run the model on SageMaker.
- Scikit-learn for its preprocessing and machine learning tools.
- XGBoost for the efficient gradient boosting algorithm used in training.
=======
