import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import math
from torch.nn import functional as F
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Import the VAE model
from vae import VAE

# Import curve and DANN modules
from curve_generate import plot_tensile_curve
from dann_extractor import extract_features

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

# Hyperparameters for the latent predictor
input_dim = 128  # 128-dim DANN features
latent_dim = 20  # Latent space dimension from VAE
hidden_dim = 100  # Hidden layer size for the predictor
learning_rate = 1e-3
batch_size = 32
epochs = 1000
save_interval = 10

# Directories
model_save_dir = "./saved_latent_predictor_models"
loss_save_dir = "./latent_predictor_loss_records"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(loss_save_dir, exist_ok=True)

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
vae_model.eval()  # Freeze for encoding only
print(f"Pre-trained VAE loaded from {vae_pretrained_path}")

# Load data from Excel: updated_SLM-datasets.xlsx
excel_path = 'curve-com.xlsx'
if not os.path.exists(excel_path):
    raise FileNotFoundError(f"Excel file not found: {excel_path}")

data = pd.read_excel(excel_path, sheet_name=0, engine='openpyxl')

# Composition columns: Reorder to Ti, Mo, Nb, Zr, Sn, Ta
comp_cols = ['Ti/wt', 'Mo/wt', 'Nb/wt', 'Sn/wt', 'Ta/wt', 'Zr/wt']
reorder_indices = [0, 1, 2, 5, 3, 4]  # Ti, Mo, Nb, Zr, Sn, Ta
comp_data = data[comp_cols].values
comp_data_reordered = np.take(comp_data, reorder_indices, axis=1)

# Key points: 6 performance parameters
key_cols = ['Elastic modulus/GPa', 'Yield strength/MPa', 'Tensile strength/MPa', 'epsilon_uts', 'sigma_f', 'e']
key_data = data[key_cols].values

print(f"Loaded data: {comp_data_reordered.shape[0]} samples")
print(f"Composition shape: {comp_data_reordered.shape}")
print(f"Key performance shape: {key_data.shape}")

# Convert composition to tensor
comp_tensor = torch.tensor(comp_data_reordered, dtype=torch.float32)
perf_tensor = torch.tensor(key_data, dtype=torch.float32)  # Keep for evaluation

# Encode compositions to latent space using fixed VAE (use mu for deterministic encoding)
print("Encoding compositions to latent space...")
vae_model.eval()
with torch.no_grad():
    comps_to_device = comp_tensor.to(device)
    mu, _ = vae_model.encode(comps_to_device)
    latents = mu.cpu()  # (N, latent_dim)
print(f"Latent encodings shape: {latents.shape}")

# Extract DANN features from key performance points
print("Extracting DANN features from tensile curves...")
dann_checkpoint_path = 'dann_checkpoint_epoch_95.pth'
features_list = []
for i in tqdm(range(len(key_data)), desc="Generating curves and extracting features"):
    row = key_data[i]
    E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f = row
    features = generate_curve_and_extract_features(
        E, sigma_y, sigma_uts, epsilon_uts, sigma_f, epsilon_f,
        n=0.25, has_hardening=True, curvature_factor=3, smoothing_sigma=1, num_points=5000,
        checkpoint_path=dann_checkpoint_path, device=device
    )
    features_list.append(features)

features_np = np.array(features_list)
print(f"Extracted features shape: {features_np.shape} (expected: ({len(key_data)}, 128))")

if features_np.shape[1] != input_dim:
    raise ValueError(f"Feature dimension mismatch: got {features_np.shape[1]}, expected {input_dim}")

# Normalize features to match z scale (N(0,1))
print("Normalizing features to match z scale (N(0,1))...")
features_mean = np.mean(features_np, axis=0)
features_std = np.std(features_np, axis=0) + 1e-8  # Avoid division by zero
features_normalized = (features_np - features_mean) / features_std

# Save normalization parameters
np.save(os.path.join(loss_save_dir, 'features_mean.npy'), features_mean)
np.save(os.path.join(loss_save_dir, 'features_std.npy'), features_std)
print(f"Features normalized. Mean shape: {features_mean.shape}, Std shape: {features_std.shape}")

# Convert to tensor
features_tensor = torch.tensor(features_normalized, dtype=torch.float32)

# Dataset: input=DANN features (128-dim), target=latent (20-dim)
dataset = TensorDataset(features_tensor, latents)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Latent Predictor MLP (DANN features -> latent)
class LatentPredictor(nn.Module):
    def __init__(self, input_dim: int = input_dim, output_dim: int = latent_dim, hidden_dim: int = 100):
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

# Initialize the predictor model
predictor = LatentPredictor().to(device)

# Optimizer
optimizer = optim.Adam(predictor.parameters(), lr=learning_rate, weight_decay=1e-5)

# Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Training function
def train_predictor(model, dataloader, optimizer, epochs, save_interval, model_save_dir, scheduler, vae_model, features_tensor, comp_tensor, device):
    model.train()
    vae_model.eval()  # Ensure VAE remains frozen

    # Loss records
    loss_records = {
        'epoch': [],
        'mse_loss': [],
        'learning_rate': [],
        'epoch_time': []
    }

    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_start_time = time.time()

        total_epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            features, latents_batch = batch
            features = features.to(device)
            latents_batch = latents_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_latents = model(features)

            # MSE loss
            loss = nn.functional.mse_loss(pred_latents, latents_batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress
            progress_bar.set_postfix({'MSE': f"{loss.item():.4f}"})

            total_epoch_loss += loss.item()

        # Average loss
        num_batches = len(dataloader)
        avg_loss = total_epoch_loss / num_batches
        loss_records['epoch'].append(epoch + 1)
        loss_records['mse_loss'].append(avg_loss)
        loss_records['learning_rate'].append(optimizer.param_groups[0]['lr'])
        loss_records['epoch_time'].append(time.time() - epoch_start_time)

        # Print summary
        print(f"\nEpoch [{epoch + 1}/{epochs}] - Time: {loss_records['epoch_time'][-1]:.2f}s")
        print(f"MSE Loss: {avg_loss:.4f}")

        # Save model periodically
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            model_path = os.path.join(model_save_dir, f"latent_predictor_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, model_path)
            print(f"Model saved to {model_path}")

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(model_save_dir, "latent_predictor_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss
            }, best_model_path)
            print(f"New best model saved to {best_model_path}")

        # Scheduler step
        scheduler.step(avg_loss)

    return loss_records

# Save results function
def save_results(model, loss_records, model_save_dir, loss_save_dir, vae_model, features_tensor, comp_tensor, latents, perf_tensor, device):
    # Save final model
    final_model_path = os.path.join(model_save_dir, "latent_predictor_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save loss records
    loss_df = pd.DataFrame(loss_records)
    loss_csv_path = os.path.join(loss_save_dir, "training_losses.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Loss records saved to {loss_csv_path}")

    # Save config
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }
    config_df = pd.DataFrame([config])
    config_df.to_csv(os.path.join(loss_save_dir, "training_config.csv"), index=False)
    print(f"Training configuration saved")

    # Save latent encodings corresponding to compositions (as numpy for easy loading)
    latents_np = latents.numpy()
    np.save(os.path.join(loss_save_dir, "composition_latent_encodings.npy"), latents_np)
    print(f"Latent encodings saved to {os.path.join(loss_save_dir, 'composition_latent_encodings.npy')}")

    # Save as CSV for readability (with composition, performance key points, and latent)
    comp_col_names = ['Ti/wt', 'Mo/wt', 'Nb/wt', 'Zr/wt', 'Sn/wt', 'Ta/wt']  # Reordered
    perf_col_names = ['Elastic modulus/GPa', 'Yield strength/MPa', 'Tensile strength/MPa', 'epsilon_uts', 'sigma_f', 'e']
    latent_col_names = [f'Latent_{i}' for i in range(latent_dim)]

    latent_csv_data = {}
    for i, col in enumerate(comp_col_names):
        latent_csv_data[col] = comp_tensor[:, i].numpy()
    for i, col in enumerate(perf_col_names):
        latent_csv_data[col] = perf_tensor[:, i].numpy()
    for i, col in enumerate(latent_col_names):
        latent_csv_data[col] = latents_np[:, i]

    latent_df = pd.DataFrame(latent_csv_data)
    latent_csv_path = os.path.join(loss_save_dir, "composition_performance_latent_encodings.csv")
    latent_df.to_csv(latent_csv_path, index=False)
    print(f"Composition, performance key points, and latent encodings saved to {latent_csv_path}")

    # Evaluate: Predict latents from DANN features, decode to composition, save original vs reconstructed
    print("Evaluating: Reconstructing compositions via DANN features -> latent -> decode...")
    model.eval()
    vae_model.eval()
    with torch.no_grad():
        features_to_device = features_tensor.to(device)
        pred_latents = model(features_to_device)  # (N, 20)
        decoded_comps = vae_model.decode(pred_latents).cpu().numpy()  # (N, 6)

    # Results DataFrame for reconstruction
    results_data = {}
    for i, col in enumerate(comp_col_names):
        results_data[f'Original_{col}'] = comp_tensor[:, i].numpy()
        results_data[f'Reconstructed_{col}'] = decoded_comps[:, i]

    for i, col in enumerate(perf_col_names):
        results_data[f'Performance_{col}'] = perf_tensor[:, i].numpy()

    results_df = pd.DataFrame(results_data)

    # Save
    recon_csv_path = os.path.join(loss_save_dir, "original_vs_reconstructed_composition.csv")
    results_df.to_csv(recon_csv_path, index=False)
    print(f"Original, reconstructed composition, and performance key points saved to {recon_csv_path}")

# Run training
print("Starting training of DANN features -> latent predictor...")
start_time = time.time()
loss_records = train_predictor(predictor, dataloader, optimizer, epochs, save_interval, model_save_dir, scheduler, vae_model, features_tensor, comp_tensor, device)
total_time = time.time() - start_time
print(f"\nTraining completed! Total time: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.2f}s")

save_results(predictor, loss_records, model_save_dir, loss_save_dir, vae_model, features_tensor, comp_tensor, latents, perf_tensor, device)