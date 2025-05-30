import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import arch
from tqdm import tqdm
from deap import base, creator, tools, algorithms

# ====================== PLOT STYLE SETUP ======================
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ====================== DATA LOADING & PREPROCESSING ======================
print("Loading and preprocessing data...")
data = pd.read_csv(r'complete dataset') # complete dataset

# Separate features and targets
features = data.drop(['conver', 'selec'], axis=1)
labels = data[['conver', 'selec']]

# Create separate pipelines for features and labels
features_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

labels_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Apply preprocessing
features_prepared = features_pipeline.fit_transform(features)
labels_prepared = labels_pipeline.fit_transform(labels)

# Split into train/test sets
features_train, features_test, label1_train, label1_test = train_test_split(
    features_prepared, labels_prepared[:, 0], test_size=0.2, random_state=70)
_, _, label2_train, label2_test = train_test_split(
    features_prepared, labels_prepared[:, 1], test_size=0.2, random_state=70)

# Get feature names (exactly as they appear in the dataset)
feature_names = features.columns.tolist()
print("\nAvailable features in dataset:")
print(feature_names)

# ====================== MODEL DEFINITION & TRAINING ======================
print("\nTraining models...")

# Gradient Boosting Models
gbr_conver = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.38, max_depth=10,
    max_features=0.9, min_samples_split=11, min_samples_leaf=10, random_state=75)
gbr_selec = GradientBoostingRegressor(
    n_estimators=170, learning_rate=0.53, max_depth=7,
    max_features=7, min_samples_split=4, min_samples_leaf=7, random_state=79)

# Random Forest Models
rf_conver = RandomForestRegressor(n_estimators=100, max_depth=13, random_state=50)
rf_selec = RandomForestRegressor(n_estimators=89, max_depth=19, random_state=51)

# XGBoost Models
xgb_conver = XGBRegressor(n_estimators=100, learning_rate=0.90, max_depth=7, random_state=53)
xgb_selec = XGBRegressor(n_estimators=100, learning_rate=0.66, max_depth=7, random_state=51)

# Train all models
gbr_conver.fit(features_train, label1_train)
gbr_selec.fit(features_train, label2_train)
rf_conver.fit(features_train, label1_train)
rf_selec.fit(features_train, label2_train)
xgb_conver.fit(features_train, label1_train)
xgb_selec.fit(features_train, label2_train)


# ====================== VISUALIZATION FUNCTIONS ======================
def plot_feature_importance(model, feature_names, target_name, model_type):
    """Plot feature importance for a given model"""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature', y='Importance', data=importance_df,
                hue='Feature', palette='viridis', legend=False, dodge=False)
    plt.title(f'{model_type} Feature Importance ({target_name})')
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs(f"{model_type}_Plots", exist_ok=True)
    plt.savefig(f"{model_type}_Plots/{model_type}_{target_name}_Feature_Importance.png")
    importance_df.to_csv(f"{model_type}_Plots/{model_type}_{target_name}_Feature_Importance.csv", index=False)
    plt.close()


def plot_pearson_correlation(model, X_train, y_train, X_test, y_test, target_name, model_type):
    """Plot Pearson correlation between actual and predicted values"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_r, _ = stats.pearsonr(y_train, y_train_pred)
    test_r, _ = stats.pearsonr(y_test, y_test_pred)

    combined_df = pd.concat([
        pd.DataFrame({'Type': 'Train', 'Actual': y_train, 'Predicted': y_train_pred}),
        pd.DataFrame({'Type': 'Test', 'Actual': y_test, 'Predicted': y_test_pred})
    ])

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Actual', y='Predicted', hue='Type', data=combined_df,
                    palette={'Train': 'blue', 'Test': 'red'}, alpha=0.7)
    plt.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
             [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())], 'k--')
    plt.title(f'{model_type} Pearson Correlation ({target_name})\nTrain r = {train_r:.3f} | Test r = {test_r:.3f}')
    plt.legend()
    os.makedirs(f"{model_type}_Plots", exist_ok=True)
    plt.savefig(f"{model_type}_Plots/{model_type}_{target_name}_Pearson_Correlation.png")
    combined_df.to_csv(f"{model_type}_Plots/{model_type}_{target_name}_Pearson_Data.csv", index=False)
    plt.close()


def export_pdp_data(model, X_data, feature_names, target_name, model_type="XGBoost", features_of_interest=None,
                    n_samples=200):
    """Export PDP data for specific features to CSV"""
    if features_of_interest is None:
        features_of_interest = ['SSA', 'DP', 'TPV', 'D_M', 'Temp', 'Time', 'C_wt', 'P', 'SV']

    try:
        # Sample data for faster computation
        X_sample = pd.DataFrame(X_data, columns=feature_names).sample(
            n=min(n_samples, X_data.shape[0]), random_state=42)

        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Create directory for CSV exports
        csv_dir = f"{model_type}_PDP_CSV"
        os.makedirs(csv_dir, exist_ok=True)

        # Export PDP data for each feature of interest
        for feature in features_of_interest:
            if feature in feature_names:
                idx = feature_names.index(feature)
                feature_values = X_sample[feature]
                feature_shap = shap_values[:, idx] if len(shap_values.shape) > 1 else shap_values

                # Create DataFrame with the data
                pdp_data = pd.DataFrame({
                    feature: feature_values,
                    'SHAP_Value': feature_shap
                }).sort_values(by=feature)

                # Save to CSV
                csv_filename = f"{model_type}_{target_name}_PDP_{feature}.csv"
                pdp_data.to_csv(os.path.join(csv_dir, csv_filename), index=False)

        print(f"PDP data exported to CSV files in: {csv_dir}")
    except Exception as e:
        print(f"Error exporting PDP data: {str(e)}")


def shap_analysis(model, X_data, feature_names, target_name, model_type="XGBoost", n_samples=200):
    """Perform SHAP analysis and create visualizations"""
    try:
        # Sample data for SHAP (faster computation)
        X_sample = pd.DataFrame(X_data, columns=feature_names).sample(
            n=min(n_samples, X_data.shape[0]), random_state=42)

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary - {model_type} ({target_name})')

        # Save SHAP plot
        os.makedirs(f"{model_type}_SHAP", exist_ok=True)
        plt.savefig(f"{model_type}_SHAP/{model_type}_{target_name}_SHAP_Summary.png", bbox_inches='tight')
        plt.close()

        # Individual feature dependence plots
        features_of_interest = ['SSA', 'DP', 'TPV', 'D_M', 'Temp', 'Time', 'C_wt', 'P', 'SV']
        for feature in features_of_interest:
            if feature in feature_names:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature,
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    show=False
                )
                plt.title(f"SHAP Dependence: {feature} ({target_name})")
                plt.tight_layout()
                plt.savefig(f"{model_type}_SHAP/{model_type}_{target_name}_SHAP_{feature}.png", bbox_inches='tight')
                plt.close()

        # Export PDP data to CSV
        export_pdp_data(model, X_data, feature_names, target_name, model_type, features_of_interest, n_samples)

    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")


# region genetic optimization

def multi_objective_optimization(
        sample_path,
        model_conver,
        model_selec,
        features_pipeline,
        labels_pipeline,
        feature_names,
        optimize_vars,
        engineering_constraints,
        pop_size=200, # original 200
        ngen=500, # original 500
        output_dir="MOEA_Results"
):
    # ====================== initialization settings ======================
    os.makedirs(output_dir, exist_ok=True)

    # upload dataset
    sample_data = pd.read_csv(sample_path).iloc[0].to_dict()
    all_evaluations = []  # record all the results

    # ====================== initialization settings for DEAP framework ======================
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # setting for maximize both targets
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    # ====================== population initialization with constraint ======================
    var_bounds = [engineering_constraints[var] for var in optimize_vars]

    def init_gene():
        return [np.random.uniform(low, high) for (low, high) in var_bounds]

    toolbox.register("individual", tools.initIterate, creator.Individual, init_gene)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ====================== evaluation functions ======================
    def evaluate(ind):
        # constraint settings
        for i, (low, high) in enumerate(var_bounds):
            ind[i] = np.clip(ind[i], low, high)

        # building features
        feature_dict = sample_data.copy()
        for var, val in zip(optimize_vars, ind):
            feature_dict[var] = val

        # dataset normalization and prediction
        features_df = pd.DataFrame([feature_dict])[feature_names]
        scaled_features = features_pipeline.transform(features_df)
        conver_norm = float(model_conver.predict(scaled_features)[0])
        selec_norm = float(model_selec.predict(scaled_features)[0])
        all_evaluations.append({
            "variables": ind.copy(),
            "conver_norm": conver_norm,
            "selec_norm": selec_norm
        })

        return conver_norm, selec_norm

    # ====================== setting for GA algorithm ======================
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=[b[0] for b in var_bounds],
                     up=[b[1] for b in var_bounds], eta=20.0) # original eta = 20
    toolbox.register("mutate", tools.mutPolynomialBounded, low=[b[0] for b in var_bounds],
                     up=[b[1] for b in var_bounds], eta=20.0, indpb=0.6) # original eta = 20 indpb = 0.1
    toolbox.register("select", tools.selNSGA2)

    # ====================== conduct optimization ======================
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()  # maintain the Pareto front

    # setting for progress bar
    pbar = tqdm(total=ngen, desc="porcess of the GA")
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)

    # Customize the evolution process to support progress bar updates
    for gen in range(ngen):
        pop = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, pop)
        for fit, ind in zip(fits, pop):
            ind.fitness.values = fit
        hof.update(pop)
        pop = toolbox.select(pop, k=len(pop))
        pbar.update(1)
    pbar.close()

    # ====================== data processing and saving  ======================
    # include pipeline
    scaler = labels_pipeline.named_steps['std_scaler']

    # handle the pareto frontier data
    if hof.items:
        pareto_norm = np.array([ind.fitness.values for ind in hof])
        pareto_original = scaler.inverse_transform(pareto_norm)  # reverse normalization using pipeline
        variables = np.array([ind for ind in hof])

        # saving the normalized data
        pareto_df = pd.DataFrame(
            data=np.hstack([variables, pareto_norm, pareto_original]),
            columns=optimize_vars + ["Conversion_norm", "Selectivity_norm", "Conversion", "Selectivity"]
        )
        pareto_df.to_csv(os.path.join(output_dir, "pareto_front.csv"), index=False)

    # targets data processing
    if all_evaluations:
        targets_norm = np.array([[e["conver_norm"], e["selec_norm"]] for e in all_evaluations])
        targets_original = scaler.inverse_transform(targets_norm)

        variables = np.array([e["variables"] for e in all_evaluations])
        full_df = pd.DataFrame(
            data=np.hstack([variables, targets_norm, targets_original]),
            columns=optimize_vars + ["Conversion_norm", "Selectivity_norm", "Conversion", "Selectivity"]
        )
        full_df.to_csv(os.path.join(output_dir, "all_samples.csv"), index=False)

    # ====================== visual analysis ======================
    plt.figure(figsize=(10, 6))
    if all_evaluations:
        plt.scatter(targets_original[:, 0], targets_original[:, 1],
                    alpha=0.3, c='blue', label='all samples')
    if hof.items:
        plt.scatter(pareto_original[:, 0], pareto_original[:, 1],
                    edgecolors='red', facecolors='none',
                    linewidths=1.5, label='Pareto frontier')
    plt.xlabel("conversion (%)", fontsize=12)
    plt.ylabel("selectivity (%)", fontsize=12)
    plt.title("results of the GA", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "optimization_scatter.png"), dpi=300)
    plt.close()

    return hof, full_df


# main function
if __name__ == "__main__":
    # setting of the contraint
    engineering_constraints = {
        'Loading': (0, 30),
        'SSA': (100, 1000),
        'DP': (10, 100),
        'TPV': (0, 1),
        'D_M': (0, 20),
        'Temp': (200, 500),
        'Time': (0, 1000),
        'P': (0, 10.0),
        'C_wt': (0, 2000),
        'SV': (0, 15)
    }

    hof, best = multi_objective_optimization(
        sample_path=r'specific sample for optimization', # specific sample for optimization
        model_conver=xgb_conver,
        model_selec=xgb_selec,
        features_pipeline=features_pipeline,
        labels_pipeline=labels_pipeline,
        feature_names=feature_names,
        optimize_vars=['Loading', 'SSA', 'DP', 'TPV', 'D_M', 'Temp','Time', 'P', 'C_wt', 'SV'],
        engineering_constraints=engineering_constraints,
        pop_size=50,
        ngen=100,
        output_dir="MOEA_Results"
    )


# ====================== MODEL EVALUATION ======================
class ModelEvaluator:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels

    def rmse(self):
        predictions = self.model.predict(self.features)
        return np.sqrt(mean_squared_error(self.labels, predictions))

    def r2(self):
        predictions = self.model.predict(self.features)
        return r2_score(self.labels, predictions)


def evaluate_model(model, name, X_train, y_train, X_test, y_test):
    print(f"\nEvaluating {name}...")

    # Train set evaluation
    train_eval = ModelEvaluator(model, X_train, y_train)
    train_rmse = train_eval.rmse()
    train_r2 = train_eval.r2()

    # Test set evaluation
    test_eval = ModelEvaluator(model, X_test, y_test)
    test_rmse = test_eval.rmse()
    test_r2 = test_eval.r2()

    return {
        "Model": name,
        "Train RMSE": train_rmse,
        "Train R2": train_r2,
        "Test RMSE": test_rmse,
        "Test R2": test_r2
    }


# Evaluate all models
print("\nEvaluating models...")
models = {
    "GBR (conver)": gbr_conver,
    "GBR (selec)": gbr_selec,
    "RF (conver)": rf_conver,
    "RF (selec)": rf_selec,
    "XGBoost (conver)": xgb_conver,
    "XGBoost (selec)": xgb_selec
}

results = []
for name, model in models.items():
    if "conver" in name:
        results.append(evaluate_model(model, name, features_train, label1_train, features_test, label1_test))
    else:
        results.append(evaluate_model(model, name, features_train, label2_train, features_test, label2_test))

results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:")
print(results_df)


# ====================== GENERATE ALL PLOTS ======================
print("\nGenerating visualizations...")

# Create output directories
for dir_name in ["GBR_Plots", "RF_Plots", "XGBoost_Plots", "XGBoost_SHAP", "XGBoost_PDP_CSV", "GA_Optimization"]:
    os.makedirs(dir_name, exist_ok=True)

# Generate plots for each model type
model_groups = [
    ('GBR', gbr_conver, gbr_selec),
    ('RF', rf_conver, rf_selec),
    ('XGBoost', xgb_conver, xgb_selec)
]

for model_type, conver_model, selec_model in model_groups:
    print(f"\nGenerating {model_type} plots...")
    plot_feature_importance(conver_model, feature_names, 'conver', model_type)
    plot_feature_importance(selec_model, feature_names, 'selec', model_type)
    plot_pearson_correlation(conver_model, features_train, label1_train,
                             features_test, label1_test, 'conver', model_type)
    plot_pearson_correlation(selec_model, features_train, label2_train,
                             features_test, label2_test, 'selec', model_type)

# SHAP and PDP analysis (XGBoost only)
print("\nRunning SHAP and PDP analysis for XGBoost...")
shap_analysis(xgb_conver, features_train, feature_names, 'conver')
shap_analysis(xgb_selec, features_train, feature_names, 'selec')


print("\nAll analysis completed successfully!")
print("Generated output folders:")
print(f"- GBR_Plots/ (Gradient Boosting results)")
print(f"- RF_Plots/ (Random Forest results)")
print(f"- XGBoost_Plots/ (XGBoost standard plots)")
print(f"- XGBoost_SHAP/ (XGBoost SHAP analysis)")
print(f"- XGBoost_PDP_CSV/ (XGBoost PDP data in CSV format)")
print(f"- GA_Optimization/ (Genetic algorithm optimization results)")
