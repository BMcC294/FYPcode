import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPRegressor 
from sklearn.svm import SVR
from xgboost import XGBRegressor
from apricot import FacilityLocationSelection
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, ARDRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

# DEEP KNOCKOFF GENERATOR 
def generate_deep_knockoffs(X):
    """
    Creates non-linear feature twins using MLPs. 
    High max_iter ensures the conditional distributions are well-learned.
    """
    X_tilde = np.zeros_like(X)
    n, p = X.shape
    print(f"Generating Deep Knockoffs for {p} features...")
    
    for j in range(p):
        X_others = np.delete(X, j, axis=1)
        y_target = X[:, j]
        
        scaler_inner = StandardScaler()
        X_scaled = scaler_inner.fit_transform(X_others)
        
        mlp = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=100000, random_state=42)
        mlp.fit(X_scaled, y_target)
        
        pred = mlp.predict(X_scaled)
        residual = y_target - pred
        X_tilde[:, j] = pred + np.random.permutation(residual)
        
    return X_tilde

# OUTLIER DETECTION (Multi-method Comparison)
def get_clean_data(X, y, method='iso'):
    if method == 'iqr':
        Q1, Q3 = y.quantile(0.25), y.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((y < (Q1 - 1.5 * IQR)) | (y > (Q3 + 1.5 * IQR)))
    elif method == 'iso':
        mask = IsolationForest(contamination=0.1, random_state=42).fit_predict(X) == 1
    elif method == 'lof':
        mask = LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit_predict(X) == 1
    return X[mask], y[mask]

#  MASTER PIPELINE (CASH Framework)
def run_master_pipeline(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found in current directory.")
        return

    data = pd.read_csv(csv_path)
    y_raw = np.log1p(data['Dev Time (Days)'])
    X_raw = data.drop(columns=['Dev Time (Days)'])
    
    outlier_methods = ['iqr', 'iso', 'lof']
    leaderboard = []

    for out_method in outlier_methods:
        print(f"\n--- Starting Outlier Method: {out_method.upper()} ---")
        X_clean, y_clean = get_clean_data(X_raw, y_raw, method=out_method)
        
        # Knockoff Filtering Stage
        X_knockoff_input = X_clean.values
        X_deep_tilde = generate_deep_knockoffs(X_knockoff_input)
        
        mi_real = mutual_info_regression(X_clean, y_clean)
        mi_tilde = mutual_info_regression(X_deep_tilde, y_clean)
        
        # Filter features that perform worse than their "noise twins"
        knockoff_mask = mi_real > mi_tilde
        X_passed = X_clean.iloc[:, knockoff_mask]
        if X_passed.shape[1] == 0: X_passed = X_clean 
        
        # Feature Selection Logic
        selectors = {
            'None': lambda x, target: x,
            'MutualInfo': lambda x, target: x.iloc[:, np.argsort(mutual_info_regression(x, target))[-min(8, x.shape[1]):]],
            'ARD': lambda x, target: x.iloc[:, np.where(np.abs(ARDRegression().fit(x, target).coef_) > 1e-3)[0]],
            'Submodular': lambda x, target: x.iloc[:, FacilityLocationSelection(n_samples=min(8, x.shape[1]), 
                                                                               metric='correlation').fit(x.values).ranking]
        }

        # Model Grid Configuration
        models_to_tune = {
            'OLS': (LinearRegression(), {}),
            'ElasticNet': (ElasticNet(), {'alpha': [0.1, 1.0], 'l1_ratio': [0.5, 0.9]}),
            'BayesianRidge': (BayesianRidge(), {}),
            'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [5, 10]}),
            'RandomForest': (RandomForestRegressor(n_jobs=-1), {'n_estimators': [100], 'max_depth': [10]}),
            'XGBoost': (XGBRegressor(n_jobs=-1), {'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]}),
            'SVR': (SVR(), {'C': [1, 10], 'kernel': ['rbf', 'linear']}),
            'MLP': (MLPRegressor(early_stopping=True, max_iter=100000), {
                'hidden_layer_sizes': [(50,), (100, 50)],
                'alpha': [0.0001, 0.05]
            })
        }

        for sel_name, sel_func in selectors.items():
            try:
                X_selected = sel_func(X_passed, y_clean)
                
                # Standardize Features (Vital for SVR, MLP, and Gaussian Process)
                scaler = StandardScaler()
                X_final = scaler.fit_transform(X_selected)
                
                for mod_name, (model, params) in models_to_tune.items():
                    # Define Dual-Metric Scoring for CV
                    scoring = {'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
                    
                    grid = GridSearchCV(model, params, cv=5, scoring=scoring, refit='mae', n_jobs=-1)
                    grid.fit(X_final, y_clean)
                    
                    # Extract CV Metrics
                    best_idx = grid.best_index_
                    cv_r2 = grid.cv_results_['mean_test_r2'][best_idx]
                    best_params = grid.best_params_
                    
                    best_model = grid.best_estimator_
                    y_pred = best_model.predict(X_final)
                    
                    leaderboard.append({
                        'Outlier_Remover': out_method,
                        'FS_Method': sel_name,
                        'Model': mod_name,
                        'CV_MAE': -grid.best_score_,
                        'CV_R2': cv_r2,
                        'Train_MAE': mean_absolute_error(y_clean, y_pred),
                        'Best_Params': str(best_params),
                        'Num_Features': X_selected.shape[1],
                        'Features': list(X_selected.columns)
                    })
                    print(f"Success: {out_method} | {sel_name} | {mod_name} | CV_R2: {cv_r2:.3f}")
            
            except Exception as e:
                print(f"Error in {out_method}-{sel_name}-{mod_name}: {e}")

    # Compile and Save Results
    results_df = pd.DataFrame(leaderboard).sort_values(by='CV_R2', ascending=False).reset_index(drop=True)
    results_df.to_csv('final_leaderboard_results.csv', index=False)
    return results_df

if __name__ == "__main__":
    print("Master Pipeline Initialized. Running on Motorola Jira Dataset...")
    final_results = run_master_pipeline('imputed.csv')
    print("\n--- TOP 10 PERFORMANCE CONFIGURATIONS ---")
    print(final_results.head(10))
