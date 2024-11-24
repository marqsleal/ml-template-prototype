import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV

PARAM_GRID_RF = {
    'n_estimators': [10, 50, 100],
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, None]
}

PARAM_GRID_GB = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [10, 20, 30]
}

PARAM_GRID_KNN = {'n_neighbors': [3, 5, 7, 9]}

PARAM_GRID_SVM = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

PARAM_GRID_XGB = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.6, 0.8, 1.0]
}

PARAM_GRID_ADABOOST = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'base_estimator': ['algorithm', 'estimator', 'learning_rate', 'n_estimators', 'random_state']
}

PARAM_GRID_GRADIENTBOOST = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0]
}

PARAM_GRID_KMEANS = {
    'n_clusters': [2, 3, 5, 10],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20, 30],
    'max_iter': [300, 500]
}

PARAM_GRID_DBSCAN = {
    'eps': [0.3, 0.5, 0.7, 1.0],
    'min_samples': [5, 10, 15],
    'metric': ['euclidean', 'manhattan', 'cosine']
}

PARAM_GRID_HIERARCHICAL = {
    'n_clusters': [2, 3, 5, 10],
    'linkage': ['ward', 'complete', 'average', 'single'],
    'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
}

def model_train_test(df: pd.DataFrame, target: str, test_size=0.3, random_state=42) -> list:
    X, y = df.drop(columns=target), df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def model_pre_process(df: pd.DataFrame, target: str, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list:
    df = df.drop(columns=target).copy()

    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numerical_transformer = Pipeline(
        steps = [
            ('scaler', RobustScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps = [
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed


def mlflow_up(experiment: str, url='http://127.0.0.1:5000') -> None:
    mlflow.set_tracking_uri(url)
    mlflow.set_experiment(experiment)
    mlflow.sklearn.autolog(silent=True)

def supervised_rand_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1) -> list:
    with mlflow.start_run(run_name=f'Supervised_RandomSearchCV_{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        rand_search = RandomizedSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )

        rand_search.fit(X_train, y_train)
        
        best_model = rand_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_params(rand_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {rand_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

        return run_id, best_model

def supervised_grid_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1) -> list:
    with mlflow.start_run(run_name=f'Supervised_GridSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_params(grid_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {grid_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

        return run_id, best_model

def supervised_bayesian_search_cv(model, param_grid, X_train, X_test, y_train, y_test, cv=5, n_jobs=-1, verbose=1) -> list:
    with mlflow.start_run(run_name=f'Supervised_BayesSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        bayesian_search = BayesSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        bayesian_search.fit(X_train, y_train)
        
        best_model = bayesian_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)

        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_params(bayesian_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {bayesian_search.best_params_}')
        print(f'Precisão (acurácia): {accuracy}')

        return run_id, best_model

def unsupervised_rand_search_cv(model, param_grid, X_train, cv=5, n_jobs=-1, verbose=1) -> list:
    with mlflow.start_run(run_name=f'Unsupervised_RandomSearchCV_{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        rand_search = RandomizedSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )

        rand_search.fit(X_train)        
        best_model = rand_search.best_estimator_

        mlflow.log_params(rand_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {rand_search.best_params_}')

        return run_id, best_model

def unsupervised_grid_search_cv(model, param_grid, X_train, cv=5, n_jobs=-1, verbose=1) -> list:
    with mlflow.start_run(run_name=f'Unsupervised_GridSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id

        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        grid_search.fit(X_train)
        
        best_model = grid_search.best_estimator_

        mlflow.log_params(grid_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {grid_search.best_params_}')

        return run_id, best_model

def unsupervised_bayesian_search_cv(model, param_grid, X_train, cv=5, n_jobs=-1, verbose=1) -> list:
    with mlflow.start_run(run_name=f'Unsupervised_BayesSearchCV{model.__class__.__name__}') as run:
        run_id = run.info.run_id
        
        bayesian_search = BayesSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        bayesian_search.fit(X_train)
        
        best_model = bayesian_search.best_estimator_

        mlflow.log_params(bayesian_search.best_params_)
        print(f'MLflow Run ID: {run_id}')
        print(f'Melhores parâmetros: {bayesian_search.best_params_}')

        return run_id, best_model