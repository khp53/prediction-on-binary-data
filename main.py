from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, make_scorer, mean_absolute_error, precision_recall_curve, roc_curve
from sklearn.svm import SVC

class ModelSelection:
    def __init__(self, train_data_path, test_data_path):
        self.train_data = pd.read_csv(train_data_path, delimiter="\t")
        self.test_data = pd.read_csv(test_data_path, delimiter="\t")
        self.X_train = self.train_data.iloc[:, :-1]
        self.y_train = self.train_data["label"]
        self.X_test = self.test_data.iloc[:, :]

    def read_data(self):
        pass

    def preprocess_data(self):
        pass

    def train_models(self):
        self.cv_scores = StratifiedKFold(n_splits=10)
        # Make the scorer based on f1 score
        self.scorer = 'f1'

        # Logistic Regression baseline model
        self.lr_model = LogisticRegression(solver='lbfgs', max_iter=50)
        # Do stratified cross-validation
        self.lr_stratified_cv = cross_val_score(self.lr_model, self.X_train, self.y_train, cv=self.cv_scores, scoring=self.scorer)
        # Fit for plotting
        self.lr_model.fit(self.X_train, self.y_train)
        self.mean_f1_score = np.mean(self.lr_stratified_cv)

        # Gradiant Boosting Classifier
        self.gb_params = {
            'n_estimators': [20, 30, 50],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5]
        }
        self.gb_model = GradientBoostingClassifier()
        self.gb_grid_search = GridSearchCV(self.gb_model, self.gb_params, cv=self.cv_scores, scoring=self.scorer, return_train_score=True)
        self.gb_grid_search.fit(self.X_train, self.y_train)
        self.gb_score = self.gb_grid_search.best_score_

        # SVM Classifier
        self.svm_params = {
            'C': [0.1, 1, 3],
            'kernel': ['linear', 'poly', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        self.svm_model = SVC(probability=True)
        self.svm_grid_search = GridSearchCV(self.svm_model, self.svm_params, cv=self.cv_scores, scoring=self.scorer, return_train_score=True)
        self.svm_grid_search.fit(self.X_train, self.y_train)
        self.svm_score = self.svm_grid_search.best_score_

        # Get CV results for each model
        self.cv_results = {
            'Logistic Regression': {'mean': self.lr_stratified_cv.mean(),
                                    'std': self.lr_stratified_cv.std()},
            'Gradiant Boosting': {'mean': self.gb_grid_search.cv_results_['mean_train_score'].mean(),
                             'std': self.gb_grid_search.cv_results_['mean_train_score'].std()},
            'SVM': {'mean': self.svm_grid_search.cv_results_['mean_train_score'].mean(),
                    'std': self.svm_grid_search.cv_results_['mean_train_score'].std()}
        }

        # Get the best hyperparameter settings and scores
        print("--------------------------------------------------------------------------------")
        print("Best F1 score for Logistic Regression:", self.mean_f1_score)
        print("--------------------------------------------------------------------------------")
        print("Best parameters for Gradient Boosting:", self.gb_grid_search.best_params_)
        print("Best F1 score for Gradient Boosting:", self.gb_grid_search.best_score_)
        print("--------------------------------------------------------------------------------")
        print("Best parameters for SVM:", self.svm_grid_search.best_params_)
        print("Best F1 score for SVM:", self.svm_grid_search.best_score_)
        print("--------------------------------------------------------------------------------")

    def display_cv_results(self):
        # Convert results to DataFrame for better visualization
        cv_results_df = pd.DataFrame(self.cv_results).T
        cv_results_df.index.name = 'Model'
        print("Cross-Validation Results:")
        print(cv_results_df)

    def find_best_model(self):
        # There were some other comparisons that were done in the original code
        # but decided to use Gradient Boosting as the best model
        self.best_model_name = "GradianBoosting"

    def train_final_model(self):
        # The comparison here is done for ease of use
        # Cause I tried all the models
        if self.best_model_name == 'GradianBoosting':
            self.best_model = GradientBoostingClassifier(**self.gb_grid_search.best_params_)
        elif self.best_model_name == 'LogisticRegression':
            self.best_model = self.lr_model
        else:
            self.best_model = SVC(**self.svm_grid_search.best_params_)

        self.final_model = self.best_model.fit(self.X_train, self.y_train)

    def predict(self):
        if hasattr(self.final_model, 'predict_proba'):
            self.y_pred_proba = self.final_model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.final_model, 'decision_function'):
            self.y_pred_proba = self.final_model.decision_function(self.X_test)
        else:
            raise AttributeError("Model does not have a predict_proba or decision_function method.")


    def write_predictions(self, predictions_file):
        np.savetxt(predictions_file, self.y_pred_proba, fmt='%.2f', delimiter='\n')

    # Evaluate models on precision-recall and ROC curves
    def plot_precision_recall_curve(self, model, name):
        precision, recall, _ = precision_recall_curve(self.y_train, model.predict_proba(self.X_train)[:, 1])
        plt.plot(recall, precision, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')

    def plot_roc_curve(self, model, name):
        fpr, tpr, _ = roc_curve(self.y_train, model.predict_proba(self.X_train)[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=name + ' (AUROC = %0.2f)' % roc_auc)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc='best')

    def execute(self, predictions_file):
        self.read_data()
        self.preprocess_data()
        self.train_models()
        self.display_cv_results()
        self.find_best_model()
        self.train_final_model()
        self.predict()
        self.write_predictions(predictions_file)
        
        plt.figure(figsize=(10, 5))

        # Plot precision-recall curves
        plt.subplot(1, 2, 1)
        self.plot_precision_recall_curve(self.lr_model, 'Logistic Regression')
        self.plot_precision_recall_curve(self.gb_grid_search.best_estimator_, 'Gradient Boosting')
        self.plot_precision_recall_curve(self.svm_grid_search.best_estimator_, 'SVM')

        # Random classifier
        random_pred = np.random.randint(2, size=len(self.y_train))
        precision, recall, _ = precision_recall_curve(self.y_train, random_pred)
        plt.plot(recall, precision, linestyle='--', label='Random Classifier')
        plt.legend(loc='best')

        # Plot ROC curves
        plt.subplot(1, 2, 2)
        self.plot_roc_curve(self.lr_model, 'Logistic Regression')
        self.plot_roc_curve(self.gb_grid_search.best_estimator_, 'Gradient Boosting')
        self.plot_roc_curve(self.svm_grid_search.best_estimator_, 'SVM')

        # Random classifier
        random_pred = np.random.randint(2, size=len(self.y_train))
        precision, recall, _ = roc_curve(self.y_train, random_pred)
        plt.plot(recall, precision, linestyle='--', label='Random Classifier')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

# Usage
if __name__ == "__main__":
    model_selector = ModelSelection("A3_TrainData.tsv", "A3_TestData.tsv")
    model_selector.execute("A3_predictions_202381708.txt")
