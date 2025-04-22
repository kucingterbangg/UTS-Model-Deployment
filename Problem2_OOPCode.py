# link video = https://drive.google.com/drive/folders/136ABi3Ju191lqCTVlzZG7UYQqJPCLw5P?usp=sharing

"""import the needed libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle as pkl

class DataLoader:
    """handles data loading and basic inspection"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
    
    def load_data(self):
        """load dataset from file"""
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        return self.df.head()
    
    def get_data(self):
        """return as dataframe"""
        return self.df

class DataExplorer:
    """performs exploratory data analysis"""
    def __init__(self, df):
        self.df = df
    
    def show_summary(self):
        """display statistical values and NULLs"""
        display(self.df.describe())
        print("Missing values:")
        display(self.df.isna().sum())
    
    def plot_correlations(self):
        """plot feature correlation matrix"""
        plt.figure(figsize=(10,8))
        sns.heatmap(self.df.select_dtypes(include=np.number).corr(), annot=True)
        plt.title('Correlation Matrix')
        plt.show()
    
    def plot_target_distribution(self, target_col):
        """visualize distribution of target variable"""
        target = self.df[target_col].value_counts()
        sns.barplot(x=target.index, y=target.values)
        plt.title('Loan Status Distribution')
        plt.show()

class DataPreprocessor:
    """handles data cleaning and transformation"""
    def __init__(self, df):
        self.df = df.copy()
    
    def clean_gender(self):
        """clean up gender values and encode it"""
        self.df['person_gender'] = self.df['person_gender'].replace({
            'fe male': 'female', 'Male': 'male'
        }).map({'male': 1, 'female': 0})
        return self
    
    def encode_defaults(self):
        """encode yes and no to binary"""
        self.df['previous_loan_defaults_on_file'] = self.df['previous_loan_defaults_on_file'].replace({
            'Yes': 1, 'No': 0
        })
        return self
    
    def one_hot_encode(self, columns):
        """apply one-hot encoding to categorical features"""
        for col in columns:
            dummies = pd.get_dummies(self.df[col], prefix=col)
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(columns=[col], inplace=True)
        return self
    
    def get_processed_data(self):
        """return processed dataframe"""
        return self.df

class DataSplitter:
    """for train-test splitting and scaling"""
    def __init__(self, df, target_col):
        self.X = df.drop(target_col, axis=1)
        self.y = df[target_col]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split(self, test_size=0.2, random_state=0):
        """split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y)
        return self
    
    def impute_missing(self, columns):
        """fill missing values with mean"""
        imputer = SimpleImputer(strategy='mean')
        self.X_train[columns] = imputer.fit_transform(self.X_train[columns])
        self.X_test[columns] = imputer.transform(self.X_test[columns])
        return self
    
    def scale_features(self, exclude_cols=None):
        """apply robust scaling to numerical features"""
        if exclude_cols is None:
            exclude_cols = []
            
        numerical_cols = [col for col in self.X_train.select_dtypes(include=np.number).columns 
                         if col not in exclude_cols]
        scaler = RobustScaler()
        self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])
        return self
    
    def get_split_data(self):
        """return split datasets"""
        return self.X_train, self.X_test, self.y_train, self.y_test

class ModelTrainer:
    """handles model training and evaluation"""
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None
    
    def train_xgboost(self, params=None, cv_params=None):
        """train XGBoost with grid search"""
        if params is None:
            params = {}
        if cv_params is None:
            cv_params = {
                'min_child_weight': [1, 5, 10],
                'gamma': [1, 2, 5],
                'max_depth': [3, 4, 5]
            }
        
        xgb = XGBClassifier(**params)
        grid = GridSearchCV(xgb, cv_params, n_jobs=4, verbose=2)
        grid.fit(self.X_train, self.y_train)
        
        self.model = XGBClassifier(**grid.best_params_)
        self.model.fit(self.X_train, self.y_train)
        return grid.best_params_
    
    def evaluate(self, X_test, y_test):
        """evaluate model performance"""
        train_acc = self.model.score(self.X_train, self.y_train)
        test_acc = accuracy_score(self.model.predict(X_test), y_test)
        
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        
        # Confusion matrix
        sns.heatmap(confusion_matrix(y_test, self.model.predict(X_test)), 
                    annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.show()
        
        # Classification report
        print(classification_report(y_test, self.model.predict(X_test)))
    
    def save_model(self, filename):
        """save trained model to file"""
        with open(filename, 'wb') as f:
            pkl.dump(self.model, f)
        print(f"Model saved as {filename}")

#main function
if __name__ == "__main__":
    # 1. Load data
    loader = DataLoader('Dataset_A_loan.csv')
    loader.load_data()
    raw_data = loader.get_data()
    
    # 2. Explore data
    explorer = DataExplorer(raw_data)
    explorer.show_summary()
    explorer.plot_correlations()
    explorer.plot_target_distribution('loan_status')
    
    # 3. Preprocess data
    preprocessor = DataPreprocessor(raw_data)
    processed_data = (preprocessor
                     .clean_gender()
                     .encode_defaults()
                     .one_hot_encode(['person_education', 'person_home_ownership', 'loan_intent'])
                     .get_processed_data())
    
    # 4. Split and scale
    splitter = DataSplitter(processed_data, 'loan_status')
    X_train, X_test, y_train, y_test = (splitter
                                       .split()
                                       .impute_missing(['person_income'])
                                       .scale_features(exclude_cols=['person_gender', 'previous_loan_defaults_on_file'])
                                       .get_split_data())
    
    # 5. Train and evaluate
    trainer = ModelTrainer(X_train, y_train)
    best_params = trainer.train_xgboost()
    print(f"Best parameters: {best_params}")
    trainer.evaluate(X_test, y_test)
    trainer.save_model('XGB_class.pkl')