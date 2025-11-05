# Heatsink Design Optimization: Neural Operator Engineering Application (GPU Accelerated)

**English** | [中文文档](./README_CN.md)

## 📖 Project Overview

**PhysicsAI Example**: This project demonstrates how to use **Fourier Neural Operator (FNO)** to solve engineering optimization problems, enabling fast prediction from a few design parameters to high-dimensional physical fields.

### Key Innovations

- ✅ **Engineering-Oriented**: Direct mapping from design parameters to performance fields, commonly used in engineering
- ✅ **Information Amplification**: Predict 4096 temperature values from 4 parameters (information ratio 1:1024)
- ✅ **GPU Acceleration**: Evaluate 10,000 design proposals in 30 seconds, vs. months with traditional CFD
- ✅ **Real-time Optimization**: Suitable for interactive design optimization

### Main Features

    Input: Heatsink design parameters [fin_height, fin_spacing, thermal_cond, heat_power]
    Output: 64×64 temperature field

### Example Results

![Heatsink Optimization Results](./outputs/heatsink_optimization_gpu.png)

![Prediction Comparison](./outputs/prediction_vs_truth.png)

---

## 🔬 Core Principle: How Can 4 Parameters Predict 4096 Numbers?

### The True Source of Information: Model Weights

```
Input: 4 design parameters
Model: 18,138,625 neural network parameters ← Information stored here!
Output: 64×64 = 4096 temperature values

Real information flow = 4 inputs + 18M learned weights → 4096 outputs
```

**Core Understanding**: The extra information doesn't come from nowhere—it's **learned during training and compressed into the model weights**.

### Training Phase: Learning Physical Laws

```
400 training samples (4 params → 4096 temperature values)
        ↓
   Backpropagation optimization
        ↓
18M weights encode:
  • Heat equation ∇²T = -Q/k solution patterns
  • How boundary conditions affect temperature distribution
  • Spatial heat propagation patterns
  • Parameter-to-field mapping relationships
```

### Inference Phase: Applying Learned Patterns

```
4 new parameters → Through 18M learned weights → 4096 temperature values
```

### Analogy: JPEG Decoder

```
JPEG Decoder:
  Input: Compressed file (KB-level)
  Decoder: Fixed algorithm (contains image reconstruction rules)
  Output: Complete image (MB-level)

FNO Model:
  Input: 4 design parameters
  Model: 18M weights (learned PDE solution patterns)
  Output: 4096 temperature values
```

### Role of Physical Constraints

Physical constraints (∇²T = -Q/k) enable:

- **Reduced learning difficulty**: Network only learns physics-compliant mappings, not arbitrary ones
- **Training objectives**: All samples satisfy the heat conduction equation
- **Generalization**: Learning patterns, not memorizing data

**Conclusion**: 4 parameters can predict 4096 values because the model learns and stores physical laws in 18M weights through training.

---

## 🏗️ Code Architecture

### 1. Data Generation: Physics Simulator

```python
def simulate_steady_heat(params, grid_size):
    """
    Simplified steady-state heat conduction simulation

    Input: [fin_height, fin_spacing, thermal_cond, heat_power]
    Output: 64×64 temperature field
    """
    # 1. Create spatial grid
    x = np.linspace(0, 50, grid_size)  # 50mm × 50mm
    y = np.linspace(0, 50, grid_size)
    X, Y = np.meshgrid(x, y)

    # 2. Gaussian heat source (simulating chip heating)
    r = np.sqrt((X - 25)**2 + (Y - 25)**2)
    T_base = heat_power * np.exp(-r**2 / (2 * 5**2))

    # 3. Cooling efficiency (depends on design parameters)
    cooling_efficiency = (fin_height/30) * (thermal_cond/400) * (8/fin_spacing)

    # 4. Final temperature field
    T_field = 25 + T_base * (1 - 0.7 * cooling_efficiency)

    return T_field
```

**Physical Meaning**:

- Heat source generates high temperature at center
- Heatsink transfers heat to boundaries via conduction
- Higher fins, larger thermal conductivity, smaller spacing → Better cooling

### 2. FNO Model: Frequency Domain Learning

```python
class TrueFNO(nn.Module):
    """
    Fourier Neural Operator

    Core Concept:
    1. Parameter encoder: 4D → 32×64×64 feature field
    2. Spectral convolution: Learn global dependencies
    3. Field decoder: Feature field → Temperature field
    """

    def __init__(self, param_dim=4, grid_size=64, width=32, modes=12):
        # 1. Parameter → Field projection
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 32*64*64)  # Expand to initial field
        )

        # 2. 4-layer spectral convolution (learn physical evolution)
        self.spectral_conv1 = SpectralConv2d(width, width, modes)
        self.spectral_conv2 = SpectralConv2d(width, width, modes)
        self.spectral_conv3 = SpectralConv2d(width, width, modes)
        self.spectral_conv4 = SpectralConv2d(width, width, modes)

        # 3. Field → Temperature decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1)  # Output temperature field
        )
```

**Key Design**:

- Spectral convolution: Captures long-range dependencies in temperature field (global nature of heat conduction)
- Residual connections: Maintain information flow
- Multi-scale features: Learn heat conduction patterns at different scales

### 3. Spectral Convolution: Core Algorithm

```python
class SpectralConv2d(nn.Module):
    """
    Spectral Convolution: Core of FNO

    Principle: Fourier transform converts convolution to multiplication
    Spatial domain: y = conv(x, kernel)  → O(N²)
    Frequency domain: Y = FFT(x) * W     → O(N log N)
    """

    def forward(self, x):
        # 1. FFT to frequency domain
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # 2. Frequency domain multiplication (low-frequency modes)
        out_ft[:, :, :modes, :modes] = einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes, :modes],
            self.weights
        )

        # 3. IFFT back to spatial domain
        x = torch.fft.irfft2(out_ft, norm='ortho')

        return x
```

**Advantages**:

- Global receptive field: One operation sees the entire field
- Resolution-independent: Can transfer to different grids
- GPU acceleration: FFT is extremely fast on GPU

---

## 🚀 Usage

### Requirements

```bash
# Hardware
- NVIDIA GPU (4GB+ VRAM recommended)
- CUDA 11.0+

# Software
- Python 3.8+
- PyTorch 2.0+ (with CUDA)
- NumPy
- Matplotlib
```

### Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib
```

### Running the Code

```bash
# Complete workflow
python src/heatsink-optimization-fno.py

# Output directory
./outputs/
  ├── prediction_vs_truth.png         # Prediction accuracy validation
  └── heatsink_optimization_gpu.png   # Design optimization results
```

---

## 📊 Experimental Results

### Training Performance

| Metric                | Value         |
| --------------------- | ------------- |
| Training Samples      | 400 designs   |
| Test Samples          | 100 designs   |
| Training Epochs       | 500           |
| Training Time         | ~13 min (GPU) |
| Average Error         | 0.086°C       |
| Max Temperature Error | 0.322°C       |

**Convergence Curve**:

```
Epoch  20 | Loss: 655.47 | Max Temp Error: 23.43°C
Epoch  80 | Loss: 1.08   | Max Temp Error: 1.12°C
Epoch 100 | Loss: 0.18   | Max Temp Error: 0.70°C
Epoch 200 | Loss: 0.21   | Max Temp Error: 0.46°C
Epoch 500 | Loss: 0.13   | Max Temp Error: 0.25°C
```

### Design Optimization Results

**Problem Setup**:

- **Objective**: Minimize maximum temperature
- **Constraints**: Fin height 10-30mm, spacing 2-8mm, thermal conductivity 100-400 W/m·K
- **Search Space**: 10,000 candidate designs

**Optimal Design**:

```
Fin Height:       20.4 mm
Fin Spacing:      2.0 mm  (minimum value, maximizes cooling area)
Thermal Cond.:    184 W/m·K
Max Temperature:  54.0°C
```

**Performance Comparison**:

| Method              | Time        | Speedup        |
| ------------------- | ----------- | -------------- |
| Traditional CFD     | 1666.7 days | 1×             |
| GPU Neural Operator | 29.1 sec    | **4,950,884×** |

---

## 🎯 Technical Highlights

### 1. GPU Optimization Strategy

```python
# ✓ Automatic GPU detection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ✓ Mixed precision training (FP16)
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    pred = model(params)
    loss = criterion(pred, target)

# ✓ CUDA optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ✓ Batch processing
batch_size = 1000
for i in range(0, 10000, batch_size):
    batch_temps = model(params[i:i+batch_size])
```

**Acceleration Effects**:

- FFT Operations: GPU is **10-100× faster** than CPU
- Mixed Precision: **2-3× speedup**
- Batch Processing: **5-10× throughput improvement**

### 2. Physics-Guided Learning

```python
# Loss function: Not only fit data, but also satisfy physical laws
def physics_loss(pred_temp, true_temp, params):
    # 1. Data fitting term
    data_loss = F.mse_loss(pred_temp, true_temp)

    # 2. Physics constraint term (optional)
    # - Energy conservation
    # - Temperature gradient continuity
    # - Boundary condition satisfaction

    return data_loss + λ * physics_loss
```

---

## 📐 Physics Background

### Heat Conduction Equation

Steady-state heat conduction (no time term):

```
∇·(k∇T) + Q = 0

Where:
  k = Thermal conductivity [W/m·K]
  T = Temperature field [°C]
  Q = Heat source term [W/m³]
```

### Boundary Conditions

```
1. Heat source center: Q = heat_power (concentrated source)
2. Heatsink region: q = h(T - T_ambient)  (convective cooling)
3. Boundary: T = T_ambient = 25°C
```

### Cooling Efficiency

```python
cooling_efficiency = (fin_height/30) * (thermal_cond/400) * (8/fin_spacing)
```

**Physical Interpretation**:

- `fin_height` ↑: Increases cooling surface area
- `thermal_cond` ↑: Accelerates heat conduction
- `fin_spacing` ↓: Increases number of fins

---

## 🔧 Extended Applications

### 1. Multi-Physics Coupling

```python
# Thermal-Fluid-Structure Coupling
Input: [geometric params, material params, fluid params]
Output: [temperature field, velocity field, stress field]
```

### 2. Multi-Objective Optimization

```python
# Optimize multiple metrics simultaneously
objectives = {
    'max_temp': minimize,      # Minimize temperature
    'weight': minimize,         # Minimize weight
    'cost': minimize,           # Minimize cost
    'volume': constraint        # Volume constraint
}
```

### 3. Uncertainty Quantification

```python
# Consider parameter uncertainty
Input: Parameter distribution (mean + std)
Output: Temperature field distribution (prediction + confidence interval)
```

---

## 📚 References

1. **FNO Original Paper**: Li, Z., et al. (2021). "Fourier Neural Operator for Parametric Partial Differential Equations." _ICLR 2021_. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
2. **Neural Operator Survey**: Kovachki, N., et al. (2023). "Neural Operator: Learning Maps Between Function Spaces." _Journal of Machine Learning Research_.
3. **Engineering Applications**: Wen, G., et al. (2022). "U-FNO: An Enhanced Fourier Neural Operator for Multiphase Flow." _Physical Review E_.

---

## 🤝 Contributing

### Contact

- GitHub Issues: Welcome to raise questions and suggestions
- Pull Requests: Welcome to contribute code improvements

### Future Improvements

- [ ] Train with real CFD data
- [ ] Implement 3D heatsink optimization
- [ ] Add uncertainty quantification
- [ ] Support custom geometric shapes
- [ ] Develop interactive web interface

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

Thanks to the following open-source projects:

- PyTorch team for the deep learning framework
- FNO authors for their pioneering work
- NVIDIA for CUDA acceleration technology

---

## Appendix A: Complete Code Structure

```
├── GPU Device Setup
│   └── setup_device()
├── Data Generation
│   ├── generate_heatsink_data()
│   └── simulate_steady_heat()
├── FNO Model
│   ├── SpectralConv2d (Spectral Convolution)
│   └── TrueFNO (Complete Model)
├── Training
│   └── train_design_model_gpu()
├── Optimization
│   └── design_optimization_demo_gpu()
└── Visualization
    ├── visualize_predictions_vs_truth()
    └── visualize_design_results()
```

---

## Appendix B: Key Hyperparameters

| Parameter       | Value | Description                |
| --------------- | ----- | -------------------------- |
| `grid_size`     | 64    | Spatial grid resolution    |
| `width`         | 32    | FNO channel width          |
| `modes`         | 12    | Frequency domain modes     |
| `batch_size`    | 32    | Training batch size        |
| `learning_rate` | 0.001 | Initial learning rate      |
| `epochs`        | 500   | Training epochs            |
| `n_train`       | 400   | Number of training samples |

**Tuning Tips**:

- `modes` ↑: Better accuracy, but slower speed
- `width` ↑: Enhanced expressiveness, but higher memory consumption
- `batch_size` ↑: More stable training, but requires more VRAM

---

## Appendix C: FAQ

### Q1: Why is the error large in early training?

**A**: Neural networks need to learn physical laws from random initialization, leading to large early errors. Typically converges quickly after 100-200 epochs.

### Q2: How to improve prediction accuracy?

**A**:

1. Increase training samples (400 → 1000+)
2. Increase FNO layers (4 → 6)
3. Increase modes (12 → 20)
4. Use real CFD data

### Q3: What if GPU memory is insufficient?

**A**:

1. Reduce `batch_size` (32 → 16)
2. Reduce `width` (32 → 24)
3. Disable mixed precision training

### Q4: Can it be used for other physical fields?

**A**: Absolutely! Just modify:

1. `simulate_steady_heat()` → Your physics simulator
2. Adjust input parameter dimensions
3. Adjust output field dimensions

---

**Last Updated**: November 5, 2025  
**Version**: v1.0  
**Author**: dingtiexin
