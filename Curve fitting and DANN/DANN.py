import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Added for CSV saving


# Gradient Reversal Layer for DANN
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


# DANN Model
class DANN(nn.Module):
    def __init__(self, feature_dim=128):
        super(DANN, self).__init__()
        # Feature Extractor (CNN for images)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, feature_dim)
        )
        # Domain Classifier (binary: synthetic=0, real=1)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha=0.0):
        features = self.feature_extractor(x)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)
        return features, domain_pred


# Paired Custom Dataset for Curves (Modified to handle pairs explicitly)
class PairedCurveDataset(Dataset):
    def __init__(self, synthetic_paths, real_paths, transform=None):
        self.synthetic_paths = synthetic_paths
        self.real_paths = real_paths
        self.transform = transform
        assert len(synthetic_paths) == len(real_paths), "Synthetic and real paths must match in length"

    def __len__(self):
        return len(self.synthetic_paths)

    def __getitem__(self, idx):
        # Load synthetic
        synth_img = Image.open(self.synthetic_paths[idx]).convert('RGB')
        if self.transform:
            synth_img = self.transform(synth_img)
        # Load real
        real_img = Image.open(self.real_paths[idx]).convert('RGB')
        if self.transform:
            real_img = self.transform(real_img)
        # Domains: 0 for synth, 1 for real
        synth_domain = torch.tensor(0, dtype=torch.float32).unsqueeze(0)
        real_domain = torch.tensor(1, dtype=torch.float32).unsqueeze(0)
        return synth_img, real_img, synth_domain, real_domain


# Main Training Script
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 8  # Number of pairs per batch
    num_epochs = 100
    learning_rate = 0.001
    lambda_adv = 1.0  # Weight for adversarial (domain) loss
    lambda_pair = 1.0  # NEW: Weight for pair alignment loss (MSE on features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Preparation - Paired per folder (expecting 51 folders starting with E)
    base_dir = os.getcwd()
    all_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print(f"Total subfolders found: {len(all_folders)}")

    valid_folders = [f for f in all_folders if f.startswith('E')]
    print(f"Valid folders (starting with E): {len(valid_folders)}")

    synthetic_paths = []
    real_paths = []
    folder_names = []  # For tracking pairs
    missing_info = []

    for folder in valid_folders:
        folder_path = os.path.join(base_dir, folder)
        synth_path = os.path.join(folder_path, 'generated_curve.png')
        real_path = os.path.join(folder_path, 'stress-strain.png')
        synth_exists = os.path.exists(synth_path)
        real_exists = os.path.exists(real_path)

        if synth_exists and real_exists:
            synthetic_paths.append(synth_path)
            real_paths.append(real_path)
            folder_names.append(folder)
            print(f"✅ Found complete pair in folder: {folder}")
        else:
            missing_info.append((folder, synth_exists, real_exists))
            reason = []
            if not synth_exists:
                reason.append("missing generated_curve.png")
            if not real_exists:
                reason.append("missing stress-strain.png")
            print(f"❌ Incomplete in folder {folder}: {', '.join(reason)}")

    num_pairs = len(synthetic_paths)
    print(f"\nTotal matched pairs found: {num_pairs} out of {len(valid_folders)} valid folders")

    if num_pairs == 0:
        raise ValueError("No matched synthetic-real pairs found in subfolders.")

    if missing_info:
        print("\nMissing folders details:")
        for folder, synth, real in missing_info:
            print(f"  - {folder}: synthetic={synth}, real={real}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Paired Dataset and Dataloader (no shuffle for paired processing, but can add if needed)
    dataset = PairedCurveDataset(synthetic_paths, real_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=2)  # Shuffle OK, pairs are internal

    # Model, Optimizer
    model = DANN(feature_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_domain = nn.BCELoss()
    criterion_pair = nn.MSELoss()  # NEW: For pair feature alignment

    # Training history for CSV
    training_history = []

    # Define results_dir early for checkpoints
    results_dir = os.path.join(base_dir, 'dann_results')
    os.makedirs(results_dir, exist_ok=True)

    # Training Loop with real-time printing (every epoch)
    losses = []
    domain_accuracies = []
    print("\nStarting DANN training with pair alignment...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pair_loss = 0.0
        correct_domains = 0
        total_samples = 0

        p = float(epoch) / num_epochs  # Progress for ramp-up
        lambda_p = 2. / (1. + np.exp(-20 * p)) - 1  # Steep sigmoid ramp-up (faster to 1)

        for batch_idx, (synth_imgs, real_imgs, synth_domains, real_domains) in enumerate(dataloader):
            synth_imgs = synth_imgs.to(device)
            real_imgs = real_imgs.to(device)
            synth_domains = synth_domains.to(device)
            real_domains = real_domains.to(device)
            batch_size_actual = synth_imgs.size(0)

            optimizer.zero_grad()

            # Forward for synthetic
            synth_features, synth_dom_pred = model(synth_imgs, alpha=lambda_p)
            # Forward for real
            real_features, real_dom_pred = model(real_imgs, alpha=lambda_p)

            # NEW: Pair alignment loss (MSE between matched synth and real features)
            pair_loss = criterion_pair(synth_features, real_features)
            pair_loss.backward(retain_graph=True)  # Backprop pair loss first

            # Domain loss: Concatenate predictions and domains
            all_features = torch.cat([synth_features, real_features], dim=0)
            all_dom_pred = torch.cat([synth_dom_pred, real_dom_pred], dim=0)
            all_domains = torch.cat(
                [synth_domains.expand(batch_size_actual, 1), real_domains.expand(batch_size_actual, 1)], dim=0)

            dom_loss = criterion_domain(all_dom_pred, all_domains)
            (lambda_adv * lambda_p * dom_loss).backward()  # Scaled adversarial domain loss

            # Total loss (for logging, but backprop already done)
            total_loss = dom_loss + (lambda_pair * pair_loss)

            # Accuracy for monitoring (on concatenated)
            dom_pred_binary = (all_dom_pred > 0.5).float()
            correct_domains += (dom_pred_binary == all_domains).sum().item()
            total_samples += all_domains.size(0)

            optimizer.step()
            epoch_loss += total_loss.item()
            epoch_pair_loss += pair_loss.item()

            # Print per batch
            if batch_idx % (len(dataloader) // 5) == 0:
                print(
                    f"  Batch {batch_idx}/{len(dataloader)}, Total Loss: {total_loss.item():.4f}, Pair Loss: {pair_loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        avg_pair_loss_val = epoch_pair_loss / len(dataloader)
        avg_acc = correct_domains / total_samples
        losses.append(avg_loss)
        domain_accuracies.append(avg_acc)

        # Log to history
        training_history.append({
            'epoch': epoch + 1,
            'avg_total_loss': avg_loss,
            'avg_pair_loss': avg_pair_loss_val,
            'domain_acc': avg_acc,
            'lambda_p': lambda_p
        })

        # Real-time print every epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Avg Total Loss: {avg_loss:.4f}, Avg Pair Loss: {avg_pair_loss_val:.4f}, "
            f"Domain Acc: {avg_acc:.4f}, Lambda: {lambda_p:.4f}")

        # NEW: Save intermediate model checkpoint every 10 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(results_dir, f'dann_checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

    print("Training completed!")

    # Save Model and Training Artifacts
    # results_dir already defined above

    # Save final model parameters
    torch.save(model.state_dict(), os.path.join(results_dir, 'dann_model.pth'))
    print(f"Model saved to: {os.path.join(results_dir, 'dann_model.pth')}")

    # NEW: Save training history to CSV
    df_history = pd.DataFrame(training_history)
    df_history.to_csv(os.path.join(results_dir, 'training_history.csv'), index=False)
    print(f"Training history saved to CSV: {os.path.join(results_dir, 'training_history.csv')}")

    # Plot and save training curves (updated for pair loss)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 2)
    plt.plot([h['avg_pair_loss'] for h in training_history])
    plt.title('Pair Alignment Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Pair Loss')

    plt.subplot(1, 3, 3)
    plt.plot(domain_accuracies)
    plt.title('Domain Classification Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to: {os.path.join(results_dir, 'training_curves.png')}")

    # Extract and visualize latent features (t-SNE for similarity check)
    # Also compute pairwise distances for matched pairs
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        synth_features = []
        real_features = []
        synth_labels = []
        real_labels = []
        pair_distances = []  # Euclidean distances between matched pairs

        # Synthetics
        for i in range(num_pairs):
            img_path = synthetic_paths[i]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            feat, _ = model(img, alpha=0.0)
            synth_features.append(feat.cpu().numpy().flatten())
            synth_labels.append(0)

        # Reals
        for i in range(num_pairs):
            img_path = real_paths[i]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            feat, _ = model(img, alpha=0.0)
            real_features.append(feat.cpu().numpy().flatten())
            real_labels.append(1)

        # Compute matched pair distances
        synth_feats = np.array(synth_features)
        real_feats = np.array(real_features)
        for i in range(num_pairs):
            dist = np.linalg.norm(synth_feats[i] - real_feats[i])
            pair_distances.append(dist)
        avg_pair_dist = np.mean(pair_distances)
        print(f"Average Euclidean distance in latent space for matched pairs: {avg_pair_dist:.4f}")

        # t-SNE
        all_features = np.vstack(synth_features + real_features)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
        tsne_features = tsne.fit_transform(all_features)
        labels = synth_labels + real_labels

        plt.figure(figsize=(10, 6))
        colors = ['blue' if l == 0 else 'red' for l in labels]
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=colors, s=50, alpha=0.7)
        # Connect matched pairs
        for i in range(num_pairs):
            plt.plot([tsne_features[i, 0], tsne_features[num_pairs + i, 0]],
                     [tsne_features[i, 1], tsne_features[num_pairs + i, 1]],
                     'k--', alpha=0.3, linewidth=1)
        plt.title('t-SNE of Latent Features (Synthetic: Blue, Real: Red, Dashed: Matched Pairs)')
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Synthetic'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Real')]
        plt.legend(handles=legend_elements)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'latent_tsne.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Latent space visualization saved to: {os.path.join(results_dir, 'latent_tsne.png')}")

    # Save pair distances to CSV (NEW)
    df_distances = pd.DataFrame({'pair_index': range(num_pairs), 'euclidean_distance': pair_distances})
    df_distances.to_csv(os.path.join(results_dir, 'matched_pair_distances.csv'), index=False)
    print(f"Pair distances saved to CSV: {os.path.join(results_dir, 'matched_pair_distances.csv')}")

    print("DANN training completed. Results saved to 'dann_results' folder.")