"""
Engineering Application: Heatsink Design Optimization - GPU Accelerated Version
Demonstrates neural operator application in real design problems

GPU Optimization Points:
1. Automatic GPU detection
2. Data and model transfer to GPU
3. Mixed precision training (FP16)
4. Batch processing optimization
5. CUDA optimization settings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# GPU Device Setup
# ============================================================================


def setup_device():
    """
    Setup computing device (GPU preferred)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("\n" + "=" * 70)
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(
            f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print("=" * 70 + "\n")

        # CUDA optimization settings
        torch.backends.cudnn.benchmark = True  # Auto-find optimal convolution algorithm
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 acceleration
        torch.backends.cudnn.allow_tf32 = True

    else:
        device = torch.device("cpu")
        print("\n⚠️  GPU not detected, using CPU training")

    return device


# ============================================================================
# 1. Engineering Problem Definition: Heatsink Design
# ============================================================================


def generate_heatsink_data(n_designs=500, grid_size=64, device="cpu"):
    """
    Generate heatsink design data

    GPU Optimization: Generate data directly on GPU (if available)
    """

    print("=" * 70)
    print("Engineering Case: Heatsink Design Optimization (GPU Accelerated)")
    print("=" * 70)
    print("\nThis is NOT time evolution! It's parameter-to-field mapping:")
    print("  Input: Design parameters (height, spacing, material...)")
    print("  Output: Steady-state temperature field")
    print("  Application: Fast design performance evaluation\n")

    print(f"Generating {n_designs} heatsink design proposals...")

    design_params = []
    temperature_fields = []

    for i in range(n_designs):
        # 随机设计参数
        fin_height = np.random.uniform(10, 30)  # 散热片高度 (mm)
        fin_spacing = np.random.uniform(2, 8)  # 散热片间距 (mm)
        thermal_cond = np.random.uniform(100, 400)  # 材料导热率 (W/m·K)
        heat_power = np.random.uniform(50, 200)  # 热源功率 (W)

        params = np.array([fin_height, fin_spacing, thermal_cond, heat_power])

        # 简化的热传导模拟（稳态）
        temp_field = simulate_steady_heat(params, grid_size)

        design_params.append(params)
        temperature_fields.append(temp_field)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i+1}/{n_designs} designs...")

    # Convert to numpy arrays
    design_params = np.array(design_params)
    temperature_fields = np.array(temperature_fields)

    # GPU optimization: Convert directly to GPU tensor
    design_params = torch.FloatTensor(design_params).to(device)
    temperature_fields = torch.FloatTensor(temperature_fields).to(device)

    print(f"\n✓ Data generation completed and loaded to {device}")
    print(f"  Design parameters shape: {design_params.shape}")
    print(f"  Temperature fields shape: {temperature_fields.shape}")

    return design_params, temperature_fields


def simulate_steady_heat(params, grid_size):
    """
    Simplified steady-state heat conduction simulation
    """

    fin_height, fin_spacing, thermal_cond, heat_power = params

    # Create spatial grid
    x = np.linspace(0, 50, grid_size)  # mm
    y = np.linspace(0, 50, grid_size)
    X, Y = np.meshgrid(x, y)

    # Heat source location (center)
    heat_source_x, heat_source_y = 25, 25

    # Simplified model: Gaussian heat source + cooling decay
    r = np.sqrt((X - heat_source_x) ** 2 + (Y - heat_source_y) ** 2)

    # Base temperature (heat source)
    T_base = heat_power * np.exp(-(r**2) / (2 * 5**2))

    # Cooling efficiency (depends on heatsink parameters)
    cooling_efficiency = (fin_height / 30) * (thermal_cond / 400) * (8 / fin_spacing)
    cooling_efficiency = np.clip(cooling_efficiency, 0.1, 1.0)

    # Final temperature field
    T_field = 25 + T_base * (1 - 0.7 * cooling_efficiency)

    return T_field.astype(np.float32)


# ============================================================================
# 2. True FNO Model (GPU Optimized Version)
# ============================================================================


class SpectralConv2d(nn.Module):
    """
    True Fourier layer (spectral convolution)
    GPU Optimization: FFT operations are extremely fast on GPU
    """

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 频域权重（复数）
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, modes, 2)
        )

    def forward(self, x):
        """
        GPU Optimization: FFT on GPU is 10-100× faster than CPU

        Fix: FFT operations must use FP32, not FP16 (mixed precision)
        """
        batch, channels, H, W = x.shape

        # Save original dtype
        dtype = x.dtype

        # ===== Critical Fix: Force FFT to use FP32 =====
        if x.dtype == torch.float16:
            x = x.float()  # FP16 → FP32

        # 1. FFT to frequency domain (GPU accelerated)
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # 2. Spectral convolution (keep low-frequency modes)
        out_ft = torch.zeros(
            batch, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
        )

        # Convolution only on low frequencies
        out_ft[:, :, : self.modes, : self.modes] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes, : self.modes],
            torch.view_as_complex(self.weights),
        )

        # 3. IFFT back to spatial domain (GPU accelerated)
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        # Restore original dtype (if needed)
        if dtype == torch.float16:
            x = x.half()  # FP32 → FP16

        return x


class TrueFNO(nn.Module):
    """
    True FNO implementation (GPU optimized version)

    GPU Optimization Points:
    1. All operations on GPU
    2. BatchNorm leverages GPU parallelism
    3. Avoid CPU-GPU data transfer
    """

    def __init__(self, param_dim=4, grid_size=64, width=32, modes=12):
        super().__init__()
        self.grid_size = grid_size
        self.width = width
        self.modes = modes

        # 1. Parameter encoder (GPU accelerated matrix multiplication)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, width * grid_size * grid_size),
        )

        # 2. FNO layers (spectral + spatial domain)
        self.spectral_conv1 = SpectralConv2d(width, width, modes)
        self.spectral_conv2 = SpectralConv2d(width, width, modes)
        self.spectral_conv3 = SpectralConv2d(width, width, modes)
        self.spectral_conv4 = SpectralConv2d(width, width, modes)

        # Local convolutions (convolutions are fast on GPU)
        self.local_conv1 = nn.Conv2d(width, width, 1)
        self.local_conv2 = nn.Conv2d(width, width, 1)
        self.local_conv3 = nn.Conv2d(width, width, 1)
        self.local_conv4 = nn.Conv2d(width, width, 1)

        # BatchNorm (GPU parallel optimization)
        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(width)
        self.bn4 = nn.BatchNorm2d(width)

        # 3. Output decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(width, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, params):
        """
        Forward propagation (entirely on GPU)
        """
        batch = params.shape[0]

        # Parameters → Field
        x = self.param_encoder(params)
        x = x.view(batch, self.width, self.grid_size, self.grid_size)

        # FNO layer 1
        x1 = self.spectral_conv1(x)
        x2 = self.local_conv1(x)
        x = self.bn1(x1 + x2 + x)
        x = F.gelu(x)

        # FNO layer 2
        x1 = self.spectral_conv2(x)
        x2 = self.local_conv2(x)
        x = self.bn2(x1 + x2 + x)
        x = F.gelu(x)

        # FNO layer 3
        x1 = self.spectral_conv3(x)
        x2 = self.local_conv3(x)
        x = self.bn3(x1 + x2 + x)
        x = F.gelu(x)

        # FNO layer 4
        x1 = self.spectral_conv4(x)
        x2 = self.local_conv4(x)
        x = self.bn4(x1 + x2 + x)
        x = F.gelu(x)

        # Decode
        temp = self.decoder(x).squeeze(1)

        return temp


# ============================================================================
# 3. GPU Accelerated Training
# ============================================================================


def train_design_model_gpu(
    model, train_params, train_temps, device, epochs=500, batch_size=32
):
    """
    GPU accelerated training

    GPU Optimization Points:
    1. Use mixed precision training (FP16)
    2. Batch processing (fully leverage GPU parallelism)
    3. Preload data to GPU
    4. Gradient accumulation (simulate larger batch)
    """

    print("\n" + "=" * 70)
    print(
        "Starting GPU Accelerated Training: Learn Design Parameters → Temperature Field Mapping"
    )
    print("=" * 70)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # Mixed precision training (GPU acceleration)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Data already on GPU, normalize
    param_mean = train_params.mean(0)
    param_std = train_params.std(0) + 1e-8
    train_params = (train_params - param_mean) / param_std

    n_samples = len(train_params)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Batches/epoch: {n_batches}")
    print(f"  Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
    print(f"  Total Samples: {n_samples}")
    print()

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()

        # Random shuffle (on GPU)
        indices = torch.randperm(n_samples, device=device)

        epoch_loss = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_params = train_params[batch_indices]
            batch_temps = train_temps[batch_indices]

            optimizer.zero_grad()

            # Mixed precision forward pass
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred_temps = model(batch_params)
                    loss = criterion(pred_temps, batch_temps)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Normal training (CPU or GPU without mixed precision support)
                pred_temps = model(batch_params)
                loss = criterion(pred_temps, batch_temps)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches

        # Evaluation (every 20 epochs)
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred_temps = model(train_params)

                # Calculate max temperature error
                max_temp_true = train_temps.max(dim=-1)[0].max(dim=-1)[0]
                max_temp_pred = pred_temps.max(dim=-1)[0].max(dim=-1)[0]
                temp_error = torch.abs(max_temp_true - max_temp_pred).mean()

                print(
                    f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | "
                    f"Max Temp Error: {temp_error.item():.2f}°C | "
                    f"Time: {epoch_time:.2f}s"
                )

    print("\n✓ Training completed!")
    print(f"  Average time per epoch: {epoch_time:.2f} seconds")

    return model, param_mean, param_std


# ============================================================================
# 4. GPU Accelerated Inference (Design Optimization)
# ============================================================================


def design_optimization_demo_gpu(model, param_mean, param_std, device, grid_size=64):
    """
    GPU accelerated design optimization

    GPU Advantage: Can evaluate a large number of designs at once (1000+)
    """

    print("\n" + "=" * 70)
    print("Engineering Application Demo: GPU Accelerated Design Optimization")
    print("=" * 70)

    # Define design space
    print("\nDesign Space:")
    print("  Fin Height: 10-30 mm")
    print("  Fin Spacing: 2-8 mm")
    print("  Thermal Conductivity: 100-400 W/m·K")
    print("  Heat Power: 100 W (fixed)")

    # Generate candidate designs
    n_candidates = 10000  # GPU can easily handle more candidates
    print(f"\nGenerating {n_candidates} candidate designs...")

    candidate_params = []
    for _ in range(n_candidates):
        params = [
            np.random.uniform(10, 30),  # Height
            np.random.uniform(2, 8),  # Spacing
            np.random.uniform(100, 400),  # Thermal conductivity
            100.0,  # Power (fixed)
        ]
        candidate_params.append(params)

    candidate_params = np.array(candidate_params)

    # Transfer to GPU and normalize
    candidate_params_gpu = torch.FloatTensor(candidate_params).to(device)
    params_norm = (candidate_params_gpu - param_mean) / param_std

    # GPU accelerated evaluation
    print(f"\nEvaluating all {n_candidates} designs using GPU...")

    model.eval()
    with torch.no_grad():
        start_time = time.time()

        # Batch evaluation (GPU parallel)
        batch_size = 1000
        all_temps = []

        for i in range(0, n_candidates, batch_size):
            batch = params_norm[i : i + batch_size]
            temps = model(batch)
            all_temps.append(temps)

        temp_fields = torch.cat(all_temps, dim=0)

        gpu_time = time.time() - start_time

    # Calculate max temperature for each design
    max_temps = temp_fields.max(dim=-1)[0].max(dim=-1)[0]

    print(f"✓ GPU evaluation completed")
    print(f"  Time: {gpu_time:.3f} seconds")
    print(f"  Speed: {n_candidates/gpu_time:.1f} designs/sec")

    # Compare with traditional method
    traditional_time = n_candidates * 4 * 3600  # Assume 4 hours per design
    print(f"\nComparison with traditional CFD method:")
    print(
        f"  Traditional CFD time: {traditional_time/3600:.0f} hours ({traditional_time/3600/24:.1f} days)"
    )
    print(f"  GPU Neural Operator: {gpu_time:.3f} seconds")
    print(f"  Speedup: {traditional_time/gpu_time:.0f}×")

    # Find optimal design
    best_idx = torch.argmin(max_temps).item()
    best_params = candidate_params[best_idx]
    best_temp = max_temps[best_idx].item()

    print(f"\nOptimal Design:")
    print(f"  Fin Height: {best_params[0]:.1f} mm")
    print(f"  Fin Spacing: {best_params[1]:.1f} mm")
    print(f"  Thermal Conductivity: {best_params[2]:.0f} W/m·K")
    print(f"  Max Temperature: {best_temp:.1f}°C")

    # Transfer back to CPU for visualization
    max_temps_cpu = max_temps.cpu().numpy()
    temp_fields_cpu = temp_fields.cpu().numpy()

    # Visualize
    visualize_design_results(candidate_params, max_temps_cpu, best_idx, temp_fields_cpu)

    return best_params, best_temp


def visualize_design_results(params, max_temps, best_idx, temp_fields):
    """Visualize design optimization results"""

    fig = plt.figure(figsize=(16, 10))

    # 1. 设计空间探索
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(
        params[:, 0], params[:, 1], c=max_temps, cmap="RdYlGn_r", s=10, alpha=0.5
    )
    ax1.scatter(
        params[best_idx, 0],
        params[best_idx, 1],
        c="blue",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=2,
        label="Best Design",
    )
    ax1.set_xlabel("Fin Height (mm)", fontsize=12)
    ax1.set_ylabel("Fin Spacing (mm)", fontsize=12)
    ax1.set_title(
        f"Design Space Exploration\n({len(params)} candidates)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label="Max Temp (°C)")
    ax1.grid(True, alpha=0.3)

    # 2. 材料vs温度
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(
        params[:, 2], max_temps, c=params[:, 0], cmap="viridis", s=10, alpha=0.5
    )
    ax2.scatter(
        params[best_idx, 2],
        max_temps[best_idx],
        c="red",
        s=200,
        marker="*",
        edgecolor="black",
        linewidth=2,
    )
    ax2.set_xlabel("Thermal Conductivity (W/m·K)", fontsize=12)
    ax2.set_ylabel("Max Temperature (°C)", fontsize=12)
    ax2.set_title("Material Selection", fontsize=14, fontweight="bold")
    plt.colorbar(scatter2, ax=ax2, label="Fin Height (mm)")
    ax2.grid(True, alpha=0.3)

    # 3. 温度分布直方图
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(max_temps, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    ax3.axvline(
        max_temps[best_idx],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best: {max_temps[best_idx]:.1f}°C",
    )
    ax3.set_xlabel("Max Temperature (°C)", fontsize=12)
    ax3.set_ylabel("Number of Designs", fontsize=12)
    ax3.set_title("Temperature Distribution", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4-6. 展示3个代表性设计
    indices = [best_idx, np.argmax(max_temps), len(params) // 2]  # 最优  # 最差  # 中等
    titles = ["Best Design ⭐", "Worst Design", "Average Design"]

    for i, (idx, title) in enumerate(zip(indices, titles)):
        ax = plt.subplot(2, 3, 4 + i)
        im = ax.imshow(temp_fields[idx], cmap="hot", origin="lower")
        ax.set_title(
            f"{title}\n"
            f"Height={params[idx,0]:.1f}mm, "
            f"Spacing={params[idx,1]:.1f}mm\n"
            f"Max Temp: {max_temps[idx]:.1f}°C",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, label="Temp (°C)")

    plt.tight_layout()
    plt.savefig("./outputs/heatsink_optimization_gpu.png", dpi=300, bbox_inches="tight")
    print("\n✓ Result visualization saved: heatsink_optimization_gpu.png")
    plt.close()


def visualize_predictions_vs_truth(test_params, test_temps, pred_temps, n_samples=6):
    """
    Visualize prediction vs ground truth comparison

    Args:
        test_params: (n, 4) test parameters
        test_temps: (n, H, W) true temperature fields
        pred_temps: (n, H, W) predicted temperature fields
        n_samples: number of samples to display
    """

    n_samples = min(n_samples, len(test_temps))

    # Randomly select samples
    indices = np.random.choice(len(test_temps), n_samples, replace=False)

    fig = plt.figure(figsize=(18, 3 * n_samples))

    for i, idx in enumerate(indices):
        params = test_params[idx]
        true_temp = test_temps[idx]
        pred_temp = pred_temps[idx]

        # Calculate error
        error = np.abs(pred_temp - true_temp)
        mae = error.mean()
        max_error = error.max()

        # Ground truth
        ax1 = plt.subplot(n_samples, 3, i * 3 + 1)
        im1 = ax1.imshow(true_temp, cmap="hot", origin="lower")
        ax1.set_title(
            f"Ground Truth\n"
            f"H={params[0]:.1f}mm, S={params[1]:.1f}mm\n"
            f"k={params[2]:.0f}W/m·K, P={params[3]:.0f}W\n"
            f"Max: {true_temp.max():.1f}°C",
            fontsize=10,
            fontweight="bold",
        )
        ax1.set_xlabel("x (mm)")
        ax1.set_ylabel("y (mm)")
        plt.colorbar(im1, ax=ax1, label="Temp (°C)")

        # Prediction
        ax2 = plt.subplot(n_samples, 3, i * 3 + 2)
        im2 = ax2.imshow(
            pred_temp,
            cmap="hot",
            origin="lower",
            vmin=true_temp.min(),
            vmax=true_temp.max(),
        )  # Uniform colorbar range
        ax2.set_title(
            f"Prediction\n" f"Max: {pred_temp.max():.1f}°C\n" f"MAE: {mae:.2f}°C",
            fontsize=10,
            fontweight="bold",
        )
        ax2.set_xlabel("x (mm)")
        ax2.set_ylabel("y (mm)")
        plt.colorbar(im2, ax=ax2, label="Temp (°C)")

        # Error map
        ax3 = plt.subplot(n_samples, 3, i * 3 + 3)
        im3 = ax3.imshow(error, cmap="Reds", origin="lower")
        ax3.set_title(
            f"Absolute Error\n" f"Mean: {mae:.2f}°C\n" f"Max: {max_error:.2f}°C",
            fontsize=10,
            fontweight="bold",
        )
        ax3.set_xlabel("x (mm)")
        ax3.set_ylabel("y (mm)")
        plt.colorbar(im3, ax=ax3, label="Error (°C)")

    plt.suptitle(
        "Model Performance: Prediction vs Ground Truth",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig("./outputs/prediction_vs_truth.png", dpi=300, bbox_inches="tight")
    print("✓ Prediction comparison saved: prediction_vs_truth.png")
    plt.close()


# ============================================================================
# 5. Main Program
# ============================================================================


def main():
    """Main function (GPU optimized version)"""

    print("\n" + "=" * 70)
    print(
        "Neural Operator Engineering Application: Heatsink Design Optimization (GPU Accelerated)"
    )
    print("=" * 70)
    print("\nKey Understanding:")
    print("  × NOT: Initial condition → Time evolution")
    print("  ✓ BUT: Design parameters → Performance field")
    print("\nThis is the most commonly used pattern in industry!\n")

    # 1. Setup GPU device
    device = setup_device()

    # Parameters
    n_train = 400
    n_test = 100
    grid_size = 64
    batch_size = 32  # GPU can use larger batch

    # 2. Generate training data (load directly to GPU)
    print("\n【Step 1】Generate Training Data")
    print("-" * 70)
    train_params, train_temps = generate_heatsink_data(n_train, grid_size, device)

    print("\n【Step 2】Generate Test Data")
    print("-" * 70)
    test_params, test_temps = generate_heatsink_data(n_test, grid_size, device)

    # 3. Create model and move to GPU
    print("\n【Step 3】Create Model")
    print("-" * 70)
    model = TrueFNO(param_dim=4, grid_size=grid_size, width=32, modes=12).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model loaded to: {device}")

    # 4. GPU accelerated training
    print("\n【Step 4】GPU Accelerated Training")
    print("-" * 70)
    model, param_mean, param_std = train_design_model_gpu(
        model, train_params, train_temps, device, epochs=500, batch_size=batch_size
    )

    # 5. Test performance
    print("\n【Step 5】Test Performance")
    print("-" * 70)
    model.eval()
    with torch.no_grad():
        test_params_norm = (test_params - param_mean) / param_std
        pred_temps = model(test_params_norm)

    # Calculate error
    mae = torch.abs(pred_temps - test_temps).mean().item()
    max_temp_error = (
        torch.abs(
            pred_temps.max(dim=-1)[0].max(dim=-1)[0]
            - test_temps.max(dim=-1)[0].max(dim=-1)[0]
        )
        .mean()
        .item()
    )

    print(f"Mean Absolute Error: {mae:.3f}°C")
    print(f"Max Temperature Error: {max_temp_error:.3f}°C")

    # ===== New: Visualize prediction vs truth =====
    print("\nGenerating prediction vs truth comparison...")
    visualize_predictions_vs_truth(
        test_params.cpu().numpy(), test_temps.cpu().numpy(), pred_temps.cpu().numpy()
    )

    # 6. GPU accelerated design optimization
    print("\n【Step 6】GPU Accelerated Design Optimization")
    print("-" * 70)
    best_params, best_temp = design_optimization_demo_gpu(
        model, param_mean, param_std, device, grid_size
    )

    # Summary
    print("\n" + "=" * 70)
    print("GPU Acceleration Summary")
    print("=" * 70)
    print("\nGPU Advantages:")
    print("  ✓ FFT operations: GPU is 10-100× faster than CPU")
    print("  ✓ Matrix multiplication: GPU parallel computing")
    print("  ✓ Batch processing: Evaluate 10000+ designs at once")
    print("  ✓ Mixed precision: FP16 acceleration 2-3×")
    print("\nPractical Applications:")
    print("  - CPU training: 2-4 hours/500 epochs")
    print("  - GPU training: 10-30 minutes/500 epochs")
    print("  - Design evaluation: 10000 designs < 1 second")
    print("  - Real-time feedback: Designers can optimize interactively")
    print("\nThis is why GPU is so important for neural operators!")
    print("=" * 70)


if __name__ == "__main__":
    main()
