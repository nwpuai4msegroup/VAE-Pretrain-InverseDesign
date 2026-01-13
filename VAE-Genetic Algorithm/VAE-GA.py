import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import pickle  # For saving models if needed
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import math  # For calculations
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import randint, loguniform
from sklearn.metrics import make_scorer

# Import the VAE model
from vae import VAE

# Import curve and DANN modules
from curve_generate import plot_tensile_curve
from dann_extractor import extract_features


# Define the LatentPredictor class to match training
class LatentPredictor(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 20, hidden_dim: int = 100):
        super(LatentPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Encapsulated function to generate tensile curve and extract DANN features
def generate_curve_and_extract_features(E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f,
                                        n=0.25, has_hardening=True, curvature_factor=3, smoothing_sigma=1,
                                        num_points=5000,
                                        checkpoint_path=None, device=None):
    """
    Encapsulated function to generate tensile curve figure, convert to PIL Image in memory,
    and extract DANN features without saving any files to disk.
    """
    # Step 1: Generate the figure object using plot_tensile_curve
    fig = plot_tensile_curve(
        E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f,
        n, has_hardening, curvature_factor, smoothing_sigma, num_points
    )

    # Step 2: Convert figure to PIL Image in memory (no saving to disk)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=600, bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)

    # Close the figure to free memory
    plt.close(fig)

    # Step 3: Extract features using the PIL Image
    features = extract_features(pil_img, checkpoint_path, device)

    # Optional: Close the PIL image if no longer needed
    pil_img.close()

    return features


# 定义R²评分函数（用于超参数搜索）
def r2_scorer(y_true, y_pred):
    return r2_score(y_true, y_pred)
r2_scoring = make_scorer(r2_scorer, greater_is_better=True)

# Hyperparameters (matching training)
input_dim = 128  # 128-dim DANN features
latent_dim = 20  # Latent space dimension from VAE
hidden_dim = 100  # Hidden layer size for the predictor

# Directories
model_save_dir = "./saved_latent_predictor_models"
loss_save_dir = "./latent_predictor_loss_records"
target_save_dir = "./target_GA"
metrics_dir = "./xgboost_metrics"  # 修改为xgboost专属目录
os.makedirs(target_save_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained VAE model
vae_pretrained_path = 'vae_model_final.pth'
if not os.path.exists(vae_pretrained_path):
    raise FileNotFoundError(f"Pre-trained VAE model not found: {vae_pretrained_path}")

vae_checkpoint = torch.load(vae_pretrained_path, map_location=device, weights_only=False)
vae_model = VAE(
    input_dim=6,  # Composition dimension
    latent_dim=latent_dim,
    hidden_dim=hidden_dim
).to(device)
vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
vae_model.eval()
print(f"Pre-trained VAE loaded from {vae_pretrained_path}")

# Load trained Latent Predictor
predictor_path = os.path.join(model_save_dir, "latent_predictor_final.pth")
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Trained predictor not found: {predictor_path}")

predictor_checkpoint = torch.load(predictor_path, map_location=device, weights_only=False)
predictor = LatentPredictor(input_dim=input_dim, output_dim=latent_dim, hidden_dim=hidden_dim).to(device)
predictor.load_state_dict(predictor_checkpoint['model_state_dict'])
predictor.eval()
print(f"Trained predictor loaded from {predictor_path}")

# Load normalization parameters for DANN features
features_mean_path = os.path.join(loss_save_dir, 'features_mean.npy')
features_std_path = os.path.join(loss_save_dir, 'features_std.npy')
if not (os.path.exists(features_mean_path) and os.path.exists(features_std_path)):
    raise FileNotFoundError(f"Normalization parameters not found in {loss_save_dir}")

features_mean = np.load(features_mean_path)
features_std = np.load(features_std_path)
print(f"Loaded normalization params: mean shape {features_mean.shape}, std shape {features_std.shape}")

# Prepare data for XGBoost
print("Loading and preparing data for XGBoost training...")
df = pd.read_excel('updated_SLM-datasets.xlsx')

# Recalculate E
df['E'] = df['Laser power(W)'] / (df['Laser scan speed(mm/s)'] * df['Hatch spacing(mm)'] * df['Layer thickness (mm)'])

# Features and target
feature_cols = ['Ti/wt', 'Mo/wt', 'Nb/wt', 'Sn/wt', 'Ta/wt', 'Zr/wt', 'Laser power(W)',
                'Laser scan speed(mm/s)', 'E']
target_col = 'Elastic modulus/GPa'

# Drop rows with NaN in features or target
df = df.dropna(subset=feature_cols + [target_col])
X = df[feature_cols]
y = df[target_col]

# Split data into train and test sets (once)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Hyperparameter tuning using RandomizedSearchCV on training set (XGBoost参数)
print("Performing hyperparameter tuning for XGBoost on training set...")
xgb_param_dist = {
    'n_estimators': randint(300, 700),
    'learning_rate': loguniform(0.01, 0.08),
    'max_depth': randint(4, 8),
    'subsample': np.linspace(0.7, 0.9, 3),
}
rscv = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror', random_state=1234),
    xgb_param_dist,
    n_iter=20,
    cv=5,  # 5-fold CV on training set
    scoring=r2_scoring,
    random_state=1234,
    n_jobs=-1,
    verbose=1
)
rscv.fit(X_train, y_train)
best_params = rscv.best_params_
print(f"Best hyperparameters: {best_params}")
print(f"Best CV score (R2) on training set: {rscv.best_score_:.4f}")

# Train final model with best parameters on full training set
print("Training final XGBoost model with best parameters...")
final_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=1234)
final_model.fit(X_train, y_train)

# Evaluate on train and test sets
print("Evaluating model performance...")
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'RMSE': [train_rmse, test_rmse],
    'MAE': [train_mae, test_mae],
    'R2': [train_r2, test_r2]
})
metrics_df.to_csv(os.path.join(metrics_dir, 'xgboost_performance_metrics.csv'), index=False)
print(f"Model performance metrics saved to {os.path.join(metrics_dir, 'xgboost_performance_metrics.csv')}")
print(f"Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R2: {train_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

# -------------------------- 关键修改：保存预测值与真实值 --------------------------
# 保存训练集预测结果（包含特征、真实值、预测值）
train_results = X_train.copy()
train_results['真实弹性模量(GPa)'] = y_train.values
train_results['预测弹性模量(GPa)'] = y_train_pred
train_results['误差(GPa)'] = train_results['预测弹性模量(GPa)'] - train_results['真实弹性模量(GPa)']
train_results.to_csv(os.path.join(metrics_dir, 'xgboost_train_predictions.csv'), index=False)
print(f"训练集预测结果已保存到: {os.path.join(metrics_dir, 'xgboost_train_predictions.csv')}")

# 保存测试集预测结果（包含特征、真实值、预测值）
test_results = X_test.copy()
test_results['真实弹性模量(GPa)'] = y_test.values
test_results['预测弹性模量(GPa)'] = y_test_pred
test_results['误差(GPa)'] = test_results['预测弹性模量(GPa)'] - test_results['真实弹性模量(GPa)']
test_results.to_csv(os.path.join(metrics_dir, 'xgboost_test_predictions.csv'), index=False)
print(f"测试集预测结果已保存到: {os.path.join(metrics_dir, 'xgboost_test_predictions.csv')}")
# --------------------------------------------------------------------------------

# Process parameters
laser_powers = [150, 180, 210, 240]
laser_speeds = [600, 800, 1000, 1200]
hatch_spacing = 0.08
layer_thickness = 0.03


# Composition order for VAE decode: [Ti, Mo, Nb, Zr, Sn, Ta] (assumed wt fractions 0-1)
# For XGBoost: [Ti, Mo, Nb, Sn, Ta, Zr] (wt% 0-100)
def reorder_comp_for_xgboost(comp_wt):
    """Reorder from VAE order [Ti, Mo, Nb, Zr, Sn, Ta] to XGBoost [Ti, Mo, Nb, Sn, Ta, Zr]"""
    return np.array([comp_wt[0], comp_wt[1], comp_wt[2], comp_wt[4], comp_wt[5], comp_wt[3]])


# Function to compute min modulus for a composition (wt fractions -> wt%)
def compute_min_modulus(comp, model):
    """
    Compute minimum modulus using the trained XGBoost model
    """
    # Normalize comp to sum=1
    comp = comp / np.sum(comp)
    comp_wt = comp * 100  # Convert to wt%
    # Check constraints (penalty if violated)
    mo = comp_wt[1]
    nb = comp_wt[2]
    zr = comp_wt[3]
    sn = comp_wt[4]
    ta = comp_wt[5]
    if not (0 <= mo <= 1 and 20 <= nb <= 30 and 1 <= zr <= 2.5 and 0 <= sn <= 1 and 0.5 <= ta <= 0.85):
        return 1000.0, None  # High penalty for constraint violation

    comp_xgb = reorder_comp_for_xgboost(comp_wt)

    min_mod = np.inf
    best_process = None
    for lp in laser_powers:
        for ls in laser_speeds:
            E = lp / (ls * hatch_spacing * layer_thickness)  # Energy density J/mm^3
            input_features = np.hstack([
                comp_xgb,
                [lp, ls, E]
            ]).reshape(1, -1)
            # Single model prediction
            mod_pred = model.predict(input_features)[0]
            if mod_pred < min_mod:
                min_mod = mod_pred
                best_process = (lp, ls)
    return min_mod, best_process


# Compute initial center latent and comp
print("\nComputing initial center using performance guidance...")
target_perf_center = np.array([[30, 600, 900, 15, 200, 40]])
dann_checkpoint_path = 'dann_checkpoint_epoch_95.pth'

# Generate features for center
features_center = generate_curve_and_extract_features(
    *target_perf_center[0],
    n=0.25, has_hardening=True, curvature_factor=3, smoothing_sigma=1, num_points=5000,
    checkpoint_path=dann_checkpoint_path, device=device
)
features_center = np.array([features_center])  # Shape (1, 128)

# Normalize
features_center_norm = (features_center - features_mean) / features_std
features_center_tensor = torch.tensor(features_center_norm, dtype=torch.float32).to(device)

# Predict latent
predictor.eval()
with torch.no_grad():
    latent_center = predictor(features_center_tensor).cpu().numpy()[0]

# Decode to composition
vae_model.eval()
latent_center_t = torch.tensor(latent_center, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 20)
with torch.no_grad():
    comp_center_raw = vae_model.decode(latent_center_t).cpu().numpy()[0]
comp_center = comp_center_raw / np.sum(comp_center_raw)

print(f"Initial center latent: {latent_center}")
print(f"Initial center composition (wt%): {comp_center * 100}")
min_mod_center, best_proc_center = compute_min_modulus(comp_center, final_model)
print(f"Initial center min modulus: {min_mod_center} GPa at process {best_proc_center}")

# Save initial center
initial_df = pd.DataFrame({
    **{f'latent_{i}': [latent_center[i]] for i in range(latent_dim)},
    **{f'{col}/wt': [comp_center[i]] for i, col in enumerate(['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta'])},
    'fitness': [min_mod_center]
})
initial_df.to_csv(os.path.join(target_save_dir, 'ga_initial_center.csv'), index=False)

# Initialize list to track best per generation
bests = []

# Add initial center as 'initial'
initial_best = {
    'generation': 'initial',
    **{f'latent_{i}': latent_center[i] for i in range(latent_dim)},
    **{f'{col}_wt%': comp_center[i] * 100 for i, col in enumerate(['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta'])},
    'fitness': min_mod_center,
    'laser_power': best_proc_center[0] if best_proc_center else np.nan,
    'scan_speed': best_proc_center[1] if best_proc_center else np.nan
}
bests.append(initial_best)

# Simple Genetic Algorithm in latent space
print("\nRunning Genetic Algorithm...")
pop_size = 50
num_generations = 30
mutation_rate = 0.2
mutation_sigma = 0.2  # Increased from 0.05 to 0.2 for stronger mutation
elite_size = 5  # Number of elites to carry over

# Initial population: larger scale around center for better exploration
np.random.seed(42)  # For reproducibility
population = np.random.normal(latent_center, scale=0.2, size=(pop_size, latent_dim))

# Evaluate initial population (generation 0)
fitnesses = np.zeros(pop_size)
processes = [None] * pop_size
comps_list = []
valid_count = 0
for i in range(pop_size):
    latent_i = torch.tensor(population[i], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        comp_i_raw = vae_model.decode(latent_i).cpu().numpy()[0]
        comp_i = comp_i_raw / np.sum(comp_i_raw)
    min_mod_i, proc_i = compute_min_modulus(comp_i, final_model)
    fitnesses[i] = min_mod_i
    processes[i] = proc_i
    comps_list.append(comp_i)
    if min_mod_i < 1000:
        valid_count += 1

# Per-generation best for gen 0
min_idx = np.argmin(fitnesses)
gen0_best_fitness = fitnesses[min_idx]
gen0_best_latent = population[min_idx].copy()
gen0_best_comp = comps_list[min_idx].copy()
gen0_best_process = processes[min_idx]

gen0_best = {
    'generation': 0,
    **{f'latent_{i}': gen0_best_latent[i] for i in range(latent_dim)},
    **{f'{col}_wt%': gen0_best_comp[i] * 100 for i, col in enumerate(['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta'])},
    'fitness': gen0_best_fitness,
    'laser_power': gen0_best_process[0] if gen0_best_process else np.nan,
    'scan_speed': gen0_best_process[1] if gen0_best_process else np.nan
}
bests.append(gen0_best)

# Global best starts with gen0
best_fitness = gen0_best_fitness
best_latent = gen0_best_latent.copy()
best_comp = gen0_best_comp.copy()
best_process = gen0_best_process

fitness_history = [best_fitness]
print(f"Generation 0: {valid_count}/{pop_size} valid (in constraints), Best fitness (min modulus) = {best_fitness:.2f} GPa")

# Save initial population (generation 0)
comps_np = np.array(comps_list)
pop_df = pd.DataFrame(population, columns=[f'latent_{i}' for i in range(latent_dim)])
comp_df = pd.DataFrame(comps_np, columns=['Ti/wt', 'Mo/wt', 'Nb/wt', 'Zr/wt', 'Sn/wt', 'Ta/wt'])
pop_full_df = pd.concat([pop_df, comp_df, pd.DataFrame({'fitness': fitnesses})], axis=1)
pop_full_df.to_csv(os.path.join(target_save_dir, 'ga_population_gen_0.csv'), index=False)

# Now run generations 1 to 500
for gen in range(1, num_generations + 1):
    # Selection: tournament selection (simple: sort and take top 2*elite for parents)
    sorted_idx = np.argsort(fitnesses)  # Ascending (low better)
    elites = population[sorted_idx[:elite_size]]
    parents = population[sorted_idx[:2 * pop_size // 2]]  # Top half as parents

    # Create new population
    new_pop = elites.copy()  # Carry elites
    while len(new_pop) < pop_size:
        # Select two parents randomly from top half
        p1_idx = np.random.randint(0, len(parents))
        p2_idx = np.random.randint(0, len(parents))
        p1, p2 = parents[p1_idx], parents[p2_idx]

        # Crossover: two-point crossover
        if np.random.rand() < 0.8:  # Crossover probability
            cross_points = np.sort(np.random.choice(latent_dim, 2, replace=False))
            child = np.concatenate([p1[:cross_points[0]], p2[cross_points[0]:cross_points[1]], p1[cross_points[1]:]])
        else:
            child = p1.copy()

        # Mutation
        if np.random.rand() < mutation_rate:
            child += np.random.normal(0, mutation_sigma, latent_dim)

        new_pop = np.vstack([new_pop, child])

    population = new_pop[:pop_size]  # Trim to pop_size

    # Evaluate fitness
    fitnesses = np.zeros(pop_size)
    processes = [None] * pop_size
    comps_list = []
    valid_count = 0
    for i in range(pop_size):
        latent_i = torch.tensor(population[i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            comp_i_raw = vae_model.decode(latent_i).cpu().numpy()[0]
            comp_i = comp_i_raw / np.sum(comp_i_raw)
        min_mod_i, proc_i = compute_min_modulus(comp_i, final_model)
        fitnesses[i] = min_mod_i
        processes[i] = proc_i
        comps_list.append(comp_i)
        if min_mod_i < 1000:
            valid_count += 1

    # Per-generation best
    current_min_idx = np.argmin(fitnesses)
    current_best_fitness = fitnesses[current_min_idx]
    current_best_latent = population[current_min_idx].copy()
    current_best_comp = comps_list[current_min_idx].copy()
    current_best_process = processes[current_min_idx]

    current_best = {
        'generation': gen,
        **{f'latent_{i}': current_best_latent[i] for i in range(latent_dim)},
        **{f'{col}_wt%': current_best_comp[i] * 100 for i, col in enumerate(['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta'])},
        'fitness': current_best_fitness,
        'laser_power': current_best_process[0] if current_best_process else np.nan,
        'scan_speed': current_best_process[1] if current_best_process else np.nan
    }
    bests.append(current_best)

    # Update global best if improved
    if current_best_fitness < best_fitness:
        best_fitness = current_best_fitness
        best_latent = current_best_latent.copy()
        best_comp = current_best_comp.copy()
        best_process = current_best_process

    fitness_history.append(best_fitness)
    print(f"Generation {gen}: {valid_count}/{pop_size} valid (in constraints), Best fitness (min modulus) = {best_fitness:.2f} GPa")

    # Save population for this generation
    comps_np = np.array(comps_list)
    pop_df = pd.DataFrame(population, columns=[f'latent_{i}' for i in range(latent_dim)])
    comp_df = pd.DataFrame(comps_np, columns=['Ti/wt', 'Mo/wt', 'Nb/wt', 'Zr/wt', 'Sn/wt', 'Ta/wt'])
    pop_full_df = pd.concat([pop_df, comp_df, pd.DataFrame({'fitness': fitnesses})], axis=1)
    pop_full_df.to_csv(os.path.join(target_save_dir, f'ga_population_gen_{gen}.csv'), index=False)

# Save best per generation to CSV
bests_df = pd.DataFrame(bests)
bests_df.to_csv(os.path.join(target_save_dir, 'best_per_generation.csv'), index=False)
print(f"Best per generation saved to {os.path.join(target_save_dir, 'best_per_generation.csv')}")

# Final best
print("\nGA completed!")
print(f"Best min modulus: {best_fitness:.2f} GPa")
print(f"Best composition (wt%): Ti={best_comp[0] * 100:.2f}, Mo={best_comp[1] * 100:.2f}, Nb={best_comp[2] * 100:.2f}, "
      f"Zr={best_comp[3] * 100:.2f}, Sn={best_comp[4] * 100:.2f}, Ta={best_comp[5] * 100:.2f}")
if best_process is not None:
    print(f"Best process: Laser Power={best_process[0]} W, Scan Speed={best_process[1]} mm/s")
else:
    print("No valid process found (all candidates violated constraints)")
print(f"Best latent: {best_latent}")

# Save best results
best_results = {
    'best_latent': best_latent,
    'best_comp_wt': best_comp * 100,
    'best_modulus': best_fitness,
    'best_process': best_process,
    'fitness_history': fitness_history,
    'xgboost_metrics': {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
}
with open(os.path.join(target_save_dir, 'ga_best_results.pkl'), 'wb') as f:
    pickle.dump(best_results, f)
print(f"Best GA results saved to {os.path.join(target_save_dir, 'ga_best_results.pkl')}")

# Save final best to CSV
best_df = pd.DataFrame({
    **{f'latent_{i}': [best_latent[i]] for i in range(latent_dim)},
    **{f'{col}/wt': [best_comp[i]] for i, col in enumerate(['Ti', 'Mo', 'Nb', 'Zr', 'Sn', 'Ta'])},
    'fitness': [best_fitness]
})
best_df.to_csv(os.path.join(target_save_dir, 'ga_best_final.csv'), index=False)

# Plot convergence
plt.figure(figsize=(8, 6))
plt.plot(fitness_history)
plt.title('GA Convergence: Best Min Modulus over Generations')
plt.xlabel('Generation')
plt.ylabel('Min Modulus (GPa)')
plt.grid(True)
plt.savefig(os.path.join(target_save_dir, 'ga_convergence.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Convergence plot saved to {os.path.join(target_save_dir, 'ga_convergence.png')}")

# Save XGBoost model
joblib.dump(final_model, os.path.join(metrics_dir, 'xgboost_final_model.pkl'))
print(f"Final XGBoost model saved to {os.path.join(metrics_dir, 'xgboost_final_model.pkl')}")