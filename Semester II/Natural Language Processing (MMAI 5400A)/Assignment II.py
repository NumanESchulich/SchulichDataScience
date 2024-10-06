import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# File paths
REVIEWS_FILE = 'reviews.csv'
TRAIN_FILE = 'train.csv'
VALID_FILE = 'valid.csv'


def load_and_preprocess_data():
    """
    Loads the reviews.csv file, preprocesses the data, and splits it into
    training and validation datasets, which are saved as train.csv and valid.csv.
    """
    data = pd.read_csv(REVIEWS_FILE, delimiter='\t')

    # Bin the ratings into negative (0), neutral (1), and positive (2)
    data['Sentiment'] = data['RatingValue'].apply(
        lambda x: 0 if x in [1, 2] else 1 if x == 3 else 2
    )

    # Balance the dataset by downsampling all classes to the size of the smallest class (negative)
    negative = data[data['Sentiment'] == 0]
    neutral = data[data['Sentiment'] == 1].sample(n=len(negative), random_state=42)
    positive = data[data['Sentiment'] == 2].sample(n=len(negative), random_state=42)

    # Combine the balanced classes
    balanced_data = pd.concat([negative, neutral, positive])

    # Split the data into training and validation sets
    train_data, valid_data = train_test_split(
        balanced_data, test_size=0.2, random_state=42, stratify=balanced_data['Sentiment']
    )

    # Save train.csv and valid.csv
    train_data.to_csv(TRAIN_FILE, index=False)
    valid_data.to_csv(VALID_FILE, index=False)


def evaluate(model, X_train, y_train, X_valid, y_valid):
    """
    Trains the model and evaluates its performance on the validation set.

    Args:
        model: A machine learning model.
        X_train: Training features.
        y_train: Training labels.
        X_valid: Validation features.
        y_valid: Validation labels.

    Returns:
        A dictionary containing accuracy, F1-scores, and the confusion matrix for the validation set.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred_valid = model.predict(X_valid)
    acc_valid = accuracy_score(y_valid, y_pred_valid)
    f1_valid = f1_score(y_valid, y_pred_valid, average='macro')
    class_f1_valid = f1_score(y_valid, y_pred_valid, average=None)
    conf_matrix_valid = confusion_matrix(y_valid, y_pred_valid, normalize='true')

    # Format confusion matrix to two decimal places
    conf_matrix_valid = np.around(conf_matrix_valid, decimals=2)

    # Return validation results
    return {
        'accuracy': acc_valid,
        'f1': f1_valid,
        'class_f1': class_f1_valid,
        'confusion_matrix': conf_matrix_valid
    }


def tune_logistic_regression(X_train, y_train):
    """
    Tunes hyperparameters of the Logistic Regression model using GridSearchCV.
    """
    # Define the parameter grid for logistic regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']  # liblinear supports l1, saga supports both l1 and l2
    }

    # Initialize logistic regression model
    logistic_reg = LogisticRegression(max_iter=5000)

    # Perform grid search
    grid_search = GridSearchCV(logistic_reg, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Return the best model and its parameters
    return grid_search.best_estimator_, grid_search.best_params_


def train_and_evaluate():
    """
    Trains and evaluates five different models on the training and validation sets,
    including hyperparameter tuning for logistic regression.
    """
    # Load the training data
    train_data = pd.read_csv(TRAIN_FILE)

    # Vectorize the reviews using Bag of Words
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['Review'])
    y_train = train_data['Sentiment']

    # Load the validation data
    valid_data = pd.read_csv(VALID_FILE)
    X_valid = vectorizer.transform(valid_data['Review'])
    y_valid = valid_data['Sentiment']

    # List of models to evaluate
    models = {
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }

    # Tune logistic regression
    best_logistic_model, best_params = tune_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = best_logistic_model

    # Track the top-performing model
    best_model_name = None
    best_model_result = None
    best_accuracy = 0

    # Evaluate each model
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate(model, X_train, y_train, X_valid, y_valid)

        # Update the best model if the current one has higher accuracy
        if result['accuracy'] > best_accuracy:
            best_accuracy = result['accuracy']
            best_model_name = model_name
            best_model_result = result

    # Print confusion matrices for the top-performing model
    print(f"\n===============================")
    print(f"Best Model: {best_model_name}")
    print(f"===============================")

    print(f"Accuracy: {best_model_result['accuracy']:.4f}")
    print(f"Average F1 score: {best_model_result['f1']:.4f}")

    print("\nClass-wise F1 scores:")
    print(f"Negative: {best_model_result['class_f1'][0]:.4f}")
    print(f"Neutral: {best_model_result['class_f1'][1]:.4f}")
    print(f"Positive: {best_model_result['class_f1'][2]:.4f}")

    # Print a cleaner confusion matrix as a table for the best model
    print("\nConfusion Matrix:")
    print(f"{'':<10} {'Negative':<10} {'Neutral':<10} {'Positive':<10}")
    print(
        f"{'Negative':<10} {best_model_result['confusion_matrix'][0, 0]:<10.2f} "
        f"{best_model_result['confusion_matrix'][0, 1]:<10.2f} "
        f"{best_model_result['confusion_matrix'][0, 2]:<10.2f}"
    )
    print(
        f"{'Neutral':<10} {best_model_result['confusion_matrix'][1, 0]:<10.2f} "
        f"{best_model_result['confusion_matrix'][1, 1]:<10.2f} "
        f"{best_model_result['confusion_matrix'][1, 2]:<10.2f}"
    )
    print(
        f"{'Positive':<10} {best_model_result['confusion_matrix'][2, 0]:<10.2f} "
        f"{best_model_result['confusion_matrix'][2, 1]:<10.2f} "
        f"{best_model_result['confusion_matrix'][2, 2]:<10.2f}"
    )


if __name__ == '__main__':
    load_and_preprocess_data()
    train_and_evaluate()