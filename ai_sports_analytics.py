import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load dataset function
def load_data(file_path):
    return pd.read_csv(file_path)

# Data cleaning function
def clean_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    return df

# Feature engineering function
def feature_engineering(df):
    df['total_points'] = df['goals'] + df['assists']
    df['goal_conversion_rate'] = df['goals'] / df['shots_on_target']
    return df

# Data visualization function
def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='position', hue='performance_category')
    plt.title('Player Performance by Position')
    plt.savefig('performance_by_position.png')
    plt.show()

# Model training function
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Model evaluation function
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

# Outcome prediction function
def outcome_prediction(X):
    model = joblib.load('outcome_model.pkl')
    predictions = model.predict(X)
    return predictions

# Main function to orchestrate the process
def main():
    # Load the data
    df = load_data('player_stats.csv')
    
    # Clean the data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare features and target
    X = df.drop(['performance_category'], axis=1)
    y = df['performance_category']
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    X['position'] = label_encoder.fit_transform(X['position'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the model
    joblib.dump(model, 'outcome_model.pkl')

    # Visualize the data
    visualize_data(df)
    
if __name__ == "__main__":
    main()

# Prediction on new data
def predict_new_player_performance(new_data):
    model = joblib.load('performance_model.pkl')
    processed_data = preprocess_new_data(new_data)
    return model.predict(processed_data)

# Preprocess new data for prediction
def preprocess_new_data(new_data):
    new_df = pd.DataFrame(new_data)
    new_df = clean_data(new_df)
    new_df = feature_engineering(new_df)
    new_df['position'] = LabelEncoder().fit_transform(new_df['position'])
    return new_df

# Additional analysis function
def performance_analysis(df):
    performance_by_team = df.groupby('team')['total_points'].agg(['mean', 'std']).reset_index()
    sns.barplot(data=performance_by_team, x='team', y='mean', yerr='std')
    plt.title('Mean Total Points by Team')
    plt.xticks(rotation=45)
    plt.savefig('mean_points_by_team.png')
    plt.show()

# Command-line interface for user interaction
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AI Sports Analytics')
    parser.add_argument('--load', type=str, help='Load player data csv file')
    parser.add_argument('--predict', type=str, help='New player performance data for prediction')
    args = parser.parse_args()

    if args.load:
        df = load_data(args.load)
        df_clean = clean_data(df)
        visualize_data(df_clean)
    elif args.predict:
        with open(args.predict, 'r') as f:
            new_data = json.load(f)
        predictions = predict_new_player_performance(new_data)
        print(f'Predictions: {predictions}')