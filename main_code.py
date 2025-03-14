import os
import json
from datasets import load_dataset
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from torchvision.utils import make_grid
from scipy.stats import wasserstein_distance
import torchvision.models as models
from sklearn.neighbors import KernelDensity
import random

save_dir = "..."
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#Setting seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###
# 1. Loading MNIST Dataset
###

full_mnist_dataset = load_dataset("mnist")

transform = transforms.ToTensor()
train_images = torch.stack([transform(img) for img in full_mnist_dataset["train"]["image"]])
test_images = torch.stack([transform(img) for img in full_mnist_dataset["test"]["image"]])

train_labels = torch.tensor(full_mnist_dataset["train"]["label"])
test_labels = torch.tensor(full_mnist_dataset["test"]["label"])

all_images = torch.cat((train_images, test_images))
all_labels = torch.cat((train_labels, test_labels))


###
# 2. Function to Create Imbalancing
###


def imbalance_dataset(train_labels, train_images, keep_ratio=0.5, seed=None):
    """
    Creates an imbalanced dataset by randomly reducing occurrences of certain classes
    based on a sampled probability distribution.

    Args:
        train_labels (torch.Tensor): Original dataset labels.
        train_images (torch.Tensor): Corresponding images.
        keep_ratio (float): Fraction of samples to retain per class (1.0 = fully balanced, 0.1 = highly imbalanced).
        seed (int): Set seed for reproducibility.

    Returns:
        torch.Tensor: Modified training images with imbalance.
        torch.Tensor: Modified training labels with imbalance.
        torch.Tensor: Selected indices that were kept.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    num_classes = len(torch.unique(train_labels))  
    
   
    class_keep_probs = (torch.rand(num_classes) * (1 - keep_ratio) + keep_ratio).clamp(min=0.01)

    
    selected_indices = []
    
    for digit in range(num_classes):
        digit_indices = torch.where(train_labels == digit)[0]  
        keep_prob = class_keep_probs[digit].item()  

        
        keep_mask = torch.rand(len(digit_indices)) < keep_prob
        selected_indices.append(digit_indices[keep_mask]) 

   
    selected_indices = torch.cat(selected_indices)
    
   
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]

   
    imbalanced_images = train_images[selected_indices]
    imbalanced_labels = train_labels[selected_indices]

    return imbalanced_images, imbalanced_labels, selected_indices

###
# 3. Generate Imbalanced Datasets
###

# Define imbalance levels
imbalance_levels = [1.0, 0.9, 0.7, 0.4, 0.1]  

dataset_info = {}

for keep_ratio in imbalance_levels:
    experiment_dir = os.path.join(save_dir, f"experiment_keep_ratio_{keep_ratio}")
    os.makedirs(experiment_dir, exist_ok=True)

    
    imbalanced_images, imbalanced_labels, selected_indices = imbalance_dataset(
        train_labels, train_images, keep_ratio, seed=42
    )
    
    
    unique, counts = np.unique(imbalanced_labels.numpy(), return_counts=True)
    class_distribution = dict(zip(unique.tolist(), counts.tolist()))

    
    dataset_info[f"keep_ratio_{keep_ratio}"] = {
        "total_samples": len(imbalanced_labels),
        "class_distribution": class_distribution
    }

    
    torch.save(imbalanced_labels, os.path.join(experiment_dir, "train_labels.pth"))
    torch.save(selected_indices, os.path.join(experiment_dir, "train_indices.pth"))


with open(os.path.join(save_dir, "dataset_info.json"), "w") as f:
    json.dump(dataset_info, f, indent=4)



###
# 4. Converting MNIST Images to Tensors
###

transform = transforms.ToTensor()
train_images = torch.stack([transform(img) for img in full_mnist_dataset["train"]["image"]])

###
# 5. Image Statistics
###

mean_image = train_images.mean(dim=0)
std_image = train_images.std(dim=0)

###
# 6. Computing Label Distribution
###

unique_labels, counts = torch.unique(all_labels, return_counts=True)
label_probabilities = counts.float() / all_labels.shape[0]

###
# 7. Saveing
###

torch.save(train_images, os.path.join(save_dir, "train_images.pth"))

image_stats = {
    "mean_image": mean_image.numpy().tolist(),
    "std_image": std_image.numpy().tolist(),
    "label_probabilities": label_probabilities.numpy().tolist()
}

with open(os.path.join(save_dir, "image_statistics.json"), "w") as f:
    json.dump(image_stats, f, indent=4)

print("Images and datasets saved")

###
# 8. Defining model architecture
###

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps using sinusoidal projections

    Args:
        embed_dim (int): Output dimensionality
        scale (float, optional): Scaling factor for random weights. Default is 30

    Returns:
        torch.Tensor: Encoded features of shape (batch_size, embed_dim)
    """
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """
    Fully connected (linear) layer that reshapes outputs to feature maps

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_dim, 1, 1).
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the dense layer

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Reshaped output of shape (batch_size, output_dim, 1, 1)
        """
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):

    """
    Time-dependent score-based model with U-Net architecture

    Args:
        marginal_prob_std (callable): Function to compute the marginal probability standard deviation
        channels (list, optional): List of channel sizes for each layer. Default is [32, 64, 128, 256]
        embed_dim (int, optional): Dimensionality of the time embedding. Default is 256

    Returns:
        torch.Tensor: Processed output tensor from the U-Net structure
    """

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()

        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):

        """Forward pass for ScoreNet with time conditioning

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            t (torch.Tensor): Time step tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Score-based output of shape (batch_size, 1, height, width)
        """

        embed = self.act(self.embed(t))

        h1 = self.conv1(x) + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))

        h2 = self.conv2(h1) + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        h3 = self.conv3(h2) + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))

        h4 = self.conv4(h3) + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        h = self.tconv4(h4) + self.dense5(embed)
        h = self.act(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)
        h = self.act(self.tgnorm3(h))

        h = self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)
        h = self.act(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h
    

class EMA:

    """
    Exponential Moving Average for stabilizing model training

    Args:
        model (nn.Module): The neural network model to track
        decay (float, optional): Decay factor for updating EMA weights
    """

    def __init__(self, model, decay=0.999):
        """Exponential Moving Average (EMA) of model parameters."""
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Backup current model weights and apply EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restoring original model weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


###
# 9. Defining SDE Functions
###

def marginal_prob_std(t, sigma):
    """Computing the mean and standard deviation of density function (transition probabilites)

    Args:    
        t (torch.Tensor): A vector of time steps.
        sigma (float): The noise parameter in our SDE.  

    Returns:
        torch.Tensor: The standard deviation.
    """    
    t = t.clone().detach().to(device).float()
    return torch.sqrt((sigma**(2 * t) - 1.) / (2. * np.log(sigma)))

def diffusion_coeff(t, sigma):
    """Computing the diffusion coefficient of SDE

    Args:
        t (float or torch.Tensor): A vector of time steps.
        sigma (float): The noise parameter in SDE

    Returns:
        torch.Tensor: The vector of diffusion coefficients
    """
    t = torch.tensor(t, device=device, dtype=torch.float32)
    return torch.tensor(sigma**t, device=device, dtype=torch.float32)

###
# 10. Setting Sigma
###

sigma = 25

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

###
# 11. Loss function
###

def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """Loss function

    Args:
        model: PyTorch model instance
        x: Mini-batch of training data.    
        marginal_prob_std: Function that gives the standard deviation of the perturbation kernel
        eps: A small tolerance value for numerical stability

    Returns:
        Computed loss value.
    """

    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  

    z = torch.randn_like(x)

    std = marginal_prob_std(random_t)

    perturbed_x = x + z * std[:, None, None, None]

    score = model(perturbed_x, random_t)

    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))

    return loss


###
# 12. Loading Imbalanced Dataset Dynamically
###


for keep_ratio in imbalance_levels:
    print(f"Running experiment for keep_ratio = {keep_ratio}")

    # Define paths for this specific experiment
    experiment_dir = os.path.join(save_dir, f"experiment_keep_ratio_{keep_ratio}")
    os.makedirs(experiment_dir, exist_ok=True)

    train_labels_path = os.path.join(experiment_dir, "train_labels.pth")
    train_indices_path = os.path.join(experiment_dir, "train_indices.pth")

    if not os.path.exists(train_labels_path) or not os.path.exists(train_indices_path):
        print(f"⚠️ Missing dataset for keep_ratio={keep_ratio}. Skipping this experiment.")
        continue

    train_labels = torch.load(train_labels_path)
    train_indices = torch.load(train_indices_path)

    train_images = torch.load(os.path.join(save_dir, "train_images.pth"))[train_indices]


    print(f"Loaded dataset for keep_ratio={keep_ratio} | {len(train_images)} samples")

###
# 12. Model & Optimizer
###

batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
ema = EMA(score_model, decay=0.999)

n_epochs = 100
lr = 1e-4

optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

###
# 13. Defining Predictor-Corrector Sampler for SGM
###

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               batch_size=64, 
               num_steps=500, 
               snr=0.1,                
               eps=1e-3,
               save_dir="cluster_experiment_data",
               keep_ratio=1.0):
    """Generating samples using the Predictor-Corrector method

    Args:
        score_model: Trained PyTorch model for score-based generative modeling
        marginal_prob_std: Function computing the std of perturbation kernel
        diffusion_coeff: Function computing the diffusion coefficient
        batch_size: Number of samples per batch
        num_steps: Number of discretized time steps
        snr: Signal-to-noise ratio for Langevin correction
        eps: Smallest time step for numerical stability
        save_dir: Directory to store generated samples
        keep_ratio: The dataset imbalance setting used in training

    Returns:
        Generated samples
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]

    time_steps = torch.linspace(1., eps, num_steps, device=device, dtype=torch.float32)
    step_size = time_steps[0] - time_steps[1]

    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps, desc=f"Sampling Progress (keep_ratio={keep_ratio})"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.sqrt(torch.prod(torch.tensor(x.shape[1:], dtype=torch.float32, device=device)))
            langevin_step_size = 2 * (snr * noise_norm / (grad_norm + 1e-8))**2 

            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

        x_final = x_mean

    ###
    # 14. Generated Samples
    ###

    experiment_dir = os.path.join(save_dir, f"experiment_keep_ratio_{keep_ratio}")
    os.makedirs(experiment_dir, exist_ok=True)

    sample_path = os.path.join(experiment_dir, "generated_samples.pth")
    torch.save(x_final.cpu(), sample_path)

    sampling_metadata = {
        "device": str(device),
        "keep_ratio": keep_ratio,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "snr": snr,
        "epsilon": eps,
        "sample_path": sample_path
    }

    with open(os.path.join(experiment_dir, "sampling_metadata.json"), "w") as f:
        json.dump(sampling_metadata, f, indent=4)

    print(f"PC Sampling completed for keep_ratio={keep_ratio}. Samples saved to {sample_path}")

    return x_final

###
# 15. Loop for the different Datasets
###

if __name__ == "__main__":
    for keep_ratio in imbalance_levels:

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        print(f"Running experiment for keep_ratio = {keep_ratio}")

        experiment_dir = os.path.join(save_dir, f"experiment_keep_ratio_{keep_ratio}")
        os.makedirs(experiment_dir, exist_ok=True)

        train_labels_path = os.path.join(experiment_dir, "train_labels.pth")
        train_indices_path = os.path.join(experiment_dir, "train_indices.pth")

        if not os.path.exists(train_labels_path) or not os.path.exists(train_indices_path):
            print(f"⚠️ Missing dataset for keep_ratio={keep_ratio}. Skipping this experiment.")
            continue

        train_labels = torch.load(train_labels_path).to(device)
        train_indices = torch.load(train_indices_path)
        train_images = torch.load(os.path.join(save_dir, "train_images.pth"))[train_indices].to(device)

        print(f"Loaded dataset for keep_ratio={keep_ratio} | {len(train_images)} samples")

        dataset = TensorDataset(train_images)
        data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
        optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)

        if device.type == "cuda":
            print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated()} bytes")
            print(f"CUDA Memory Cached: {torch.cuda.memory_reserved()} bytes")
        else:
            print("Running on CPU")

        dsm_losses = []
        for epoch in trange(n_epochs, desc="Training Progress"):
            avg_loss = 0.
            num_items = 0

            for x, in data_loader:
                x = x.to(device)

                loss = loss_fn(score_model, x, marginal_prob_std_fn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                ema.update()

                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

            epoch_loss = avg_loss / num_items
            dsm_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {epoch_loss:.5f}")

            torch.save(score_model.state_dict(), os.path.join(experiment_dir, f"ckpt_epoch_{epoch+1}.pth"))
            torch.save(dsm_losses, os.path.join(experiment_dir, "dsm_losses.pth"))

        ema.apply_shadow()
        torch.save(score_model.state_dict(), os.path.join(experiment_dir, "final_model_ema.pth"))
        ema.restore() 

        print(f"Training completed for keep_ratio={keep_ratio}. Results saved in {experiment_dir}.")

    experiment_dir = os.path.join(save_dir, f"experiment_keep_ratio_{keep_ratio}")
    ckpt_path = os.path.join(experiment_dir, "final_model_ema.pth")

    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    ckpt_path = os.path.join(experiment_dir, "final_model_ema.pth")
    score_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    score_model.to(device)
    score_model.eval()

    print(f"Loaded checkpoint from {ckpt_path}")

###
# 16. Generating Samples Using PC Sampler
###
    
for keep_ratio in imbalance_levels:

    ema.apply_shadow()

    sample_batch_size = 128
    num_steps = 500
    eps = 1e-3

    samples = pc_sampler(score_model, 
                         marginal_prob_std_fn,
                         diffusion_coeff_fn, 
                         batch_size=sample_batch_size, 
                         num_steps=num_steps,  
                         eps=eps,  
                         save_dir=save_dir, 
                         keep_ratio=keep_ratio)

    samples = samples.clamp(0.0, 1.0)

    sample_path = os.path.join(experiment_dir, "generated_samples.pth")
    torch.save(samples.cpu(), sample_path)

    ###
    # 17. Save Sampling Metadata
    ###

    sampling_metadata = {
        "device": str(device),
        "keep_ratio": keep_ratio,
        "batch_size": sample_batch_size,
        "num_steps": num_steps,
        "epsilon": eps,
        "sample_path": sample_path
    }

    with open(os.path.join(experiment_dir, "sampling_metadata.json"), "w") as f:
        json.dump(sampling_metadata, f, indent=4)

    print(f"Sampling completed for keep_ratio={keep_ratio}. Samples saved at {sample_path}")

ema.restore()


###
# 18. Computing Bounds for Each Dataset
###

for keep_ratio in imbalance_levels:
    print(f"Computing bound for keep_ratio = {keep_ratio}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    experiment_dir = os.path.join(save_dir, f"experiment_keep_ratio_{keep_ratio}")

    train_labels = torch.load(os.path.join(experiment_dir, "train_labels.pth"))
    train_indices = torch.load(os.path.join(experiment_dir, "train_indices.pth"))
    train_images = torch.load(os.path.join(save_dir, "train_images.pth"))[train_indices]
    generated_samples = torch.load(os.path.join(experiment_dir, "generated_samples.pth"))
    dsm_losses = torch.load(os.path.join(experiment_dir, "dsm_losses.pth"))

    test_images = torch.stack([transform(img) for img in full_mnist_dataset["test"]["image"]])
    test_dataset = torch.utils.data.TensorDataset(test_images)  
    test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=0,
    pin_memory=False, 
    )


    ###
    # 19. Estimating π(x) using KDE
    ###

    mnist_images = train_images.view(len(train_images), -1).numpy()
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(mnist_images)

    ###
    # 20. Computing d_1(π, N(0, I)) (prior distribution)
    ###

    gaussian_samples = np.random.multivariate_normal(mean=np.zeros(784), cov=np.eye(784), size=len(mnist_images))
    w1_distance_pi_gaussian = wasserstein_distance(mnist_images.flatten(), gaussian_samples.flatten())

    print(f"Wasserstein-1 distance between MNIST and Gaussian prior (keep_ratio={keep_ratio}): {w1_distance_pi_gaussian}")

    ###
    # 21. Computing DSM Loss e_nn
    ###

    dsm_loss = dsm_losses[-1]
    print(f"Loaded DSM Loss from Last Epoch (keep_ratio={keep_ratio}): {dsm_loss}")

    ###
    # 22. Computing Lipschitz Constant ||∇s_θ||∞
    ###

    def compute_lipschitz_constant(model, dataloader, num_batches=10):
        """
        Compute the Lipschitz constant ||∇s_θ||∞ over multiple batches.

        Args:
            model: The trained Score-Based Model
            dataloader: The test data loader
            num_batches: Number of batches to average over

        Returns:
            The estimated Lipschitz constant
        """
        max_norm = 0
        model.to(device)

        for i, (images,) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)
            images.requires_grad = True
            batch_time_step = torch.ones(images.shape[0], device=device)

            scores = model(images, batch_time_step)

            gradients = torch.autograd.grad(outputs=scores, inputs=images, 
                                            grad_outputs=torch.ones_like(scores), create_graph=True)[0]

            norm_value = gradients.norm(dim=1).max().item()
            norm_value = torch.clamp(torch.tensor(norm_value), max=100).item()

            max_norm = max(max_norm, norm_value)

        return max_norm


    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    ckpt = torch.load(os.path.join(experiment_dir, "final_model_ema.pth"), map_location=device)
    score_model.load_state_dict(ckpt)
    score_model.eval()

    lipschitz_const = compute_lipschitz_constant(score_model, test_loader)
    print(f"Lipschitz Constant (keep_ratio={keep_ratio}): {lipschitz_const}")

    ###
    # 23. Computing Finite Sample Error d_1(π^N, π)
    ###

    def compute_class_distribution(labels):
        unique, counts = np.unique(labels.numpy(), return_counts=True)
        return counts / counts.sum()

    full_mnist_labels = torch.cat((train_labels, test_labels))
    true_mnist_dist = compute_class_distribution(full_mnist_labels)  
    imbalanced_train_mnist_dist = compute_class_distribution(train_labels)
    finite_sample_error = wasserstein_distance(imbalanced_train_mnist_dist, true_mnist_dist)

    print(f"Finite Sample Error d_1(π^N, π) (keep_ratio={keep_ratio}): {finite_sample_error}")

    ###
    # 24. Computing Final Bound
    ###

    R = 1  
    omega = 1.0  
    T = 1.0  
    C = 1.0  

    best_dsm_loss = min(dsm_losses)
    epsilon = best_dsm_loss / 5000
    early_stopping_error = np.sqrt(epsilon)

    print(f"Early Stopping Error (keep_ratio={keep_ratio}): {early_stopping_error}")

    score_error = dsm_loss

    delta = np.min(true_mnist_dist[true_mnist_dist > 0])  
    finite_sample_term = finite_sample_error * (1 + np.abs(np.log(delta)) / np.sqrt(epsilon) + T * lipschitz_const**2)

    distributional_error_bound = (
        early_stopping_error +
        R**(3/2) * (1 + np.sqrt(lipschitz_const)) * (
            R * np.exp(-omega * T / R**2) * w1_distance_pi_gaussian + np.sqrt(score_error + C * finite_sample_term)
        )
    )

    print(f"Empirical Bound on Wasserstein-1 Distance (keep_ratio={keep_ratio}): {distributional_error_bound}")

    ###
    # 25. Computing Wasserstein-1 Distance Between Real & Generated Samples
    ###
    #(This is very unstable. I would need to generate 70.000 samples for it to be really meaningful.)

    w1_distance_pi_generated = wasserstein_distance(mnist_images.flatten(), generated_samples.flatten())
    print(f"Wasserstein-1 distance between MNIST and Generated Samples (keep_ratio={keep_ratio}): {w1_distance_pi_generated}")

    ###
    # 28. Saving Bounds & Results
    ###

    bound_results = {
        "keep_ratio": keep_ratio,
        "W1_distance_pi_gaussian": w1_distance_pi_gaussian,
        "dsm_loss": dsm_loss,
        "lipschitz_constant": lipschitz_const,
        "finite_sample_error": finite_sample_error,
        "early_stopping_error": early_stopping_error,
        "distributional_error_bound": distributional_error_bound,
        "w1_distance_pi_generated": w1_distance_pi_generated
    }

    with open(os.path.join(experiment_dir, "bound_results.json"), "w") as f:
        json.dump(bound_results, f, indent=4)

    print(f"Bound computation completed for keep_ratio={keep_ratio}. Results saved to {experiment_dir}")
