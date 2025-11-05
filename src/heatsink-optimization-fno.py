"""
工程应用案例：散热器设计优化 - GPU加速版本
展示神经算子在实际设计问题中的应用

GPU优化要点：
1. 自动检测GPU设备
2. 数据和模型移动到GPU
3. 混合精度训练（FP16）
4. 批处理优化
5. CUDA优化设置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# GPU设备设置
# ============================================================================

def setup_device():
    """
    设置计算设备（GPU优先）
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("\n" + "="*70)
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("="*70 + "\n")
        
        # CUDA优化设置
        torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32加速
        torch.backends.cudnn.allow_tf32 = True
        
    else:
        device = torch.device('cpu')
        print("\n⚠️  未检测到GPU，使用CPU训练")
    
    return device


# ============================================================================
# 1. 工程问题定义：散热器设计
# ============================================================================

def generate_heatsink_data(n_designs=500, grid_size=64, device='cpu'):
    """
    生成散热器设计数据
    
    GPU优化：直接在GPU上生成数据（如果可能）
    """
    
    print("="*70)
    print("工程案例：散热器设计优化 (GPU加速版)")
    print("="*70)
    print("\n这不是时间演化！而是参数到场的映射：")
    print("  输入：设计参数（高度、间距、材料...）")
    print("  输出：稳态温度场")
    print("  应用：快速评估设计性能\n")
    
    print(f"生成 {n_designs} 个散热器设计方案...")
    
    design_params = []
    temperature_fields = []
    
    for i in range(n_designs):
        # 随机设计参数
        fin_height = np.random.uniform(10, 30)      # 散热片高度 (mm)
        fin_spacing = np.random.uniform(2, 8)       # 散热片间距 (mm)
        thermal_cond = np.random.uniform(100, 400)  # 材料导热率 (W/m·K)
        heat_power = np.random.uniform(50, 200)     # 热源功率 (W)
        
        params = np.array([fin_height, fin_spacing, thermal_cond, heat_power])
        
        # 简化的热传导模拟（稳态）
        temp_field = simulate_steady_heat(params, grid_size)
        
        design_params.append(params)
        temperature_fields.append(temp_field)
        
        if (i + 1) % 100 == 0:
            print(f"  完成 {i+1}/{n_designs} 个设计...")
    
    # 转换为numpy数组
    design_params = np.array(design_params)
    temperature_fields = np.array(temperature_fields)
    
    # GPU优化：直接转换为GPU tensor
    design_params = torch.FloatTensor(design_params).to(device)
    temperature_fields = torch.FloatTensor(temperature_fields).to(device)
    
    print(f"\n✓ 数据生成完成并已加载到{device}")
    print(f"  设计参数形状: {design_params.shape}")
    print(f"  温度场形状: {temperature_fields.shape}")
    
    return design_params, temperature_fields


def simulate_steady_heat(params, grid_size):
    """
    简化的稳态热传导模拟
    """
    
    fin_height, fin_spacing, thermal_cond, heat_power = params
    
    # 创建网格
    x = np.linspace(0, 50, grid_size)  # mm
    y = np.linspace(0, 50, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 热源位置（中心）
    heat_source_x, heat_source_y = 25, 25
    
    # 简化模型：高斯热源 + 散热衰减
    r = np.sqrt((X - heat_source_x)**2 + (Y - heat_source_y)**2)
    
    # 基础温度（热源）
    T_base = heat_power * np.exp(-r**2 / (2 * 5**2))
    
    # 散热效果（取决于散热片参数）
    cooling_efficiency = (fin_height / 30) * (thermal_cond / 400) * (8 / fin_spacing)
    cooling_efficiency = np.clip(cooling_efficiency, 0.1, 1.0)
    
    # 最终温度场
    T_field = 25 + T_base * (1 - 0.7 * cooling_efficiency)
    
    return T_field.astype(np.float32)


# ============================================================================
# 2. True FNO模型（GPU优化版）
# ============================================================================

class SpectralConv2d(nn.Module):
    """
    真正的傅里叶层（频域卷积）
    GPU优化：FFT操作在GPU上非常快
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
        GPU优化：FFT在GPU上比CPU快10-100倍
        
        修复：FFT操作必须用FP32，不能用FP16（混合精度）
        """
        batch, channels, H, W = x.shape
        
        # 保存原始dtype
        dtype = x.dtype
        
        # ===== 关键修复：FFT强制使用FP32 =====
        if x.dtype == torch.float16:
            x = x.float()  # FP16 → FP32
        
        # 1. FFT到频域（GPU加速）
        x_ft = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 频域卷积（保留低频modes）
        out_ft = torch.zeros(batch, self.out_channels, H, W//2+1, 
                            dtype=torch.cfloat, device=x.device)
        
        # 只在低频进行卷积
        out_ft[:, :, :self.modes, :self.modes] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes, :self.modes],
            torch.view_as_complex(self.weights)
        )
        
        # 3. IFFT回空间域（GPU加速）
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        
        # 恢复原始dtype（如果需要）
        if dtype == torch.float16:
            x = x.half()  # FP32 → FP16
        
        return x


class TrueFNO(nn.Module):
    """
    真正的FNO实现（GPU优化版）
    
    GPU优化要点：
    1. 所有操作在GPU上
    2. BatchNorm利用GPU并行
    3. 避免CPU-GPU数据传输
    """
    
    def __init__(self, param_dim=4, grid_size=64, width=32, modes=12):
        super().__init__()
        self.grid_size = grid_size
        self.width = width
        self.modes = modes
        
        # 1. 参数编码器（GPU加速矩阵乘法）
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, width * grid_size * grid_size)
        )
        
        # 2. FNO层（频域 + 空间域）
        self.spectral_conv1 = SpectralConv2d(width, width, modes)
        self.spectral_conv2 = SpectralConv2d(width, width, modes)
        self.spectral_conv3 = SpectralConv2d(width, width, modes)
        self.spectral_conv4 = SpectralConv2d(width, width, modes)
        
        # 局部卷积（GPU上卷积很快）
        self.local_conv1 = nn.Conv2d(width, width, 1)
        self.local_conv2 = nn.Conv2d(width, width, 1)
        self.local_conv3 = nn.Conv2d(width, width, 1)
        self.local_conv4 = nn.Conv2d(width, width, 1)
        
        # BatchNorm（GPU并行优化）
        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(width)
        self.bn4 = nn.BatchNorm2d(width)
        
        # 3. 输出解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(width, 64, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 1)
        )
    
    def forward(self, params):
        """
        前向传播（全程GPU）
        """
        batch = params.shape[0]
        
        # 参数 → 场
        x = self.param_encoder(params)
        x = x.view(batch, self.width, self.grid_size, self.grid_size)
        
        # FNO层1
        x1 = self.spectral_conv1(x)
        x2 = self.local_conv1(x)
        x = self.bn1(x1 + x2 + x)
        x = F.gelu(x)
        
        # FNO层2
        x1 = self.spectral_conv2(x)
        x2 = self.local_conv2(x)
        x = self.bn2(x1 + x2 + x)
        x = F.gelu(x)
        
        # FNO层3
        x1 = self.spectral_conv3(x)
        x2 = self.local_conv3(x)
        x = self.bn3(x1 + x2 + x)
        x = F.gelu(x)
        
        # FNO层4
        x1 = self.spectral_conv4(x)
        x2 = self.local_conv4(x)
        x = self.bn4(x1 + x2 + x)
        x = F.gelu(x)
        
        # 解码
        temp = self.decoder(x).squeeze(1)
        
        return temp


# ============================================================================
# 3. GPU加速训练
# ============================================================================

def train_design_model_gpu(model, train_params, train_temps, device, epochs=500, batch_size=32):
    """
    GPU加速训练
    
    GPU优化要点：
    1. 使用混合精度训练（FP16）
    2. 批处理（充分利用GPU并行）
    3. 数据预加载到GPU
    4. 梯度累积（模拟更大batch）
    """
    
    print("\n" + "="*70)
    print("开始GPU加速训练：学习 设计参数 → 温度场 的映射")
    print("="*70)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    # 混合精度训练（GPU加速）
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 数据已在GPU上，标准化
    param_mean = train_params.mean(0)
    param_std = train_params.std(0) + 1e-8
    train_params = (train_params - param_mean) / param_std
    
    n_samples = len(train_params)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"\n训练配置：")
    print(f"  设备: {device}")
    print(f"  批大小: {batch_size}")
    print(f"  批次数/epoch: {n_batches}")
    print(f"  混合精度: {'启用' if scaler else '禁用'}")
    print(f"  总样本: {n_samples}")
    print()
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        
        # 随机打乱（GPU上）
        indices = torch.randperm(n_samples, device=device)
        
        epoch_loss = 0.0
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_params = train_params[batch_indices]
            batch_temps = train_temps[batch_indices]
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_temps = model(batch_params)
                    loss = criterion(pred_temps, batch_temps)
                
                # 混合精度反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通训练（CPU或不支持混合精度的GPU）
                pred_temps = model(batch_params)
                loss = criterion(pred_temps, batch_temps)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / n_batches
        
        # 评估（每20个epoch）
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred_temps = model(train_params)
                
                # 计算最高温度误差
                max_temp_true = train_temps.max(dim=-1)[0].max(dim=-1)[0]
                max_temp_pred = pred_temps.max(dim=-1)[0].max(dim=-1)[0]
                temp_error = torch.abs(max_temp_true - max_temp_pred).mean()
                
                print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | "
                      f"Max Temp Error: {temp_error.item():.2f}°C | "
                      f"Time: {epoch_time:.2f}s")
    
    print("\n✓ 训练完成！")
    print(f"  平均每epoch时间: {epoch_time:.2f}秒")
    
    return model, param_mean, param_std


# ============================================================================
# 4. GPU加速推理（设计优化）
# ============================================================================

def design_optimization_demo_gpu(model, param_mean, param_std, device, grid_size=64):
    """
    GPU加速的设计优化
    
    GPU优势：可以一次评估大量设计（1000+）
    """
    
    print("\n" + "="*70)
    print("工程应用演示：GPU加速设计优化")
    print("="*70)
    
    # 定义设计空间
    print("\n设计空间：")
    print("  散热片高度: 10-30 mm")
    print("  散热片间距: 2-8 mm")
    print("  材料导热率: 100-400 W/m·K")
    print("  热源功率: 100 W (固定)")
    
    # 生成候选设计
    n_candidates = 10000  # GPU可以轻松处理更多候选
    print(f"\n生成 {n_candidates} 个候选设计...")
    
    candidate_params = []
    for _ in range(n_candidates):
        params = [
            np.random.uniform(10, 30),    # 高度
            np.random.uniform(2, 8),      # 间距
            np.random.uniform(100, 400),  # 导热率
            100.0                         # 功率（固定）
        ]
        candidate_params.append(params)
    
    candidate_params = np.array(candidate_params)
    
    # 转移到GPU并标准化
    candidate_params_gpu = torch.FloatTensor(candidate_params).to(device)
    params_norm = (candidate_params_gpu - param_mean) / param_std
    
    # GPU加速评估
    print(f"\n使用GPU评估所有 {n_candidates} 个设计...")
    
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        
        # 批处理评估（GPU并行）
        batch_size = 1000
        all_temps = []
        
        for i in range(0, n_candidates, batch_size):
            batch = params_norm[i:i+batch_size]
            temps = model(batch)
            all_temps.append(temps)
        
        temp_fields = torch.cat(all_temps, dim=0)
        
        gpu_time = time.time() - start_time
    
    # 计算每个设计的最高温度
    max_temps = temp_fields.max(dim=-1)[0].max(dim=-1)[0]
    
    print(f"✓ GPU评估完成")
    print(f"  时间：{gpu_time:.3f}秒")
    print(f"  速度：{n_candidates/gpu_time:.1f} 设计/秒")
    
    # 对比传统方法
    traditional_time = n_candidates * 4 * 3600  # 假设每个设计4小时
    print(f"\n对比传统CFD方法：")
    print(f"  传统CFD时间：{traditional_time/3600:.0f} 小时 ({traditional_time/3600/24:.1f} 天)")
    print(f"  GPU神经算子：{gpu_time:.3f} 秒")
    print(f"  加速比：{traditional_time/gpu_time:.0f}×")
    
    # 找到最优设计
    best_idx = torch.argmin(max_temps).item()
    best_params = candidate_params[best_idx]
    best_temp = max_temps[best_idx].item()
    
    print(f"\n最优设计：")
    print(f"  散热片高度：{best_params[0]:.1f} mm")
    print(f"  散热片间距：{best_params[1]:.1f} mm")
    print(f"  材料导热率：{best_params[2]:.0f} W/m·K")
    print(f"  最高温度：{best_temp:.1f}°C")
    
    # 转回CPU用于可视化
    max_temps_cpu = max_temps.cpu().numpy()
    temp_fields_cpu = temp_fields.cpu().numpy()
    
    # 可视化
    visualize_design_results(candidate_params, max_temps_cpu, best_idx, temp_fields_cpu)
    
    return best_params, best_temp


def visualize_design_results(params, max_temps, best_idx, temp_fields):
    """可视化设计优化结果"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 设计空间探索
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(params[:, 0], params[:, 1], 
                         c=max_temps, cmap='RdYlGn_r', s=10, alpha=0.5)
    ax1.scatter(params[best_idx, 0], params[best_idx, 1], 
               c='blue', s=200, marker='*', edgecolor='black', linewidth=2,
               label='Best Design')
    ax1.set_xlabel('Fin Height (mm)', fontsize=12)
    ax1.set_ylabel('Fin Spacing (mm)', fontsize=12)
    ax1.set_title(f'Design Space Exploration\n({len(params)} candidates)', 
                 fontsize=14, fontweight='bold')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Max Temp (°C)')
    ax1.grid(True, alpha=0.3)
    
    # 2. 材料vs温度
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(params[:, 2], max_temps, c=params[:, 0], 
                          cmap='viridis', s=10, alpha=0.5)
    ax2.scatter(params[best_idx, 2], max_temps[best_idx],
               c='red', s=200, marker='*', edgecolor='black', linewidth=2)
    ax2.set_xlabel('Thermal Conductivity (W/m·K)', fontsize=12)
    ax2.set_ylabel('Max Temperature (°C)', fontsize=12)
    ax2.set_title('Material Selection', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Fin Height (mm)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 温度分布直方图
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(max_temps, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(max_temps[best_idx], color='red', linestyle='--', linewidth=2,
               label=f'Best: {max_temps[best_idx]:.1f}°C')
    ax3.set_xlabel('Max Temperature (°C)', fontsize=12)
    ax3.set_ylabel('Number of Designs', fontsize=12)
    ax3.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4-6. 展示3个代表性设计
    indices = [best_idx,  # 最优
               np.argmax(max_temps),  # 最差
               len(params)//2]  # 中等
    titles = ['Best Design ⭐', 'Worst Design', 'Average Design']
    
    for i, (idx, title) in enumerate(zip(indices, titles)):
        ax = plt.subplot(2, 3, 4+i)
        im = ax.imshow(temp_fields[idx], cmap='hot', origin='lower')
        ax.set_title(f'{title}\n'
                    f'Height={params[idx,0]:.1f}mm, '
                    f'Spacing={params[idx,1]:.1f}mm\n'
                    f'Max Temp: {max_temps[idx]:.1f}°C',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        plt.colorbar(im, ax=ax, label='Temp (°C)')
    
    plt.tight_layout()
    plt.savefig('./outputs/heatsink_optimization_gpu.png', 
                dpi=300, bbox_inches='tight')
    print('\n✓ 结果可视化已保存: heatsink_optimization_gpu.png')
    plt.close()


def visualize_predictions_vs_truth(test_params, test_temps, pred_temps, n_samples=6):
    """
    可视化预测vs真值对比
    
    参数:
        test_params: (n, 4) 测试参数
        test_temps: (n, H, W) 真实温度场
        pred_temps: (n, H, W) 预测温度场
        n_samples: 显示多少个样本
    """
    
    n_samples = min(n_samples, len(test_temps))
    
    # 随机选择样本
    indices = np.random.choice(len(test_temps), n_samples, replace=False)
    
    fig = plt.figure(figsize=(18, 3*n_samples))
    
    for i, idx in enumerate(indices):
        params = test_params[idx]
        true_temp = test_temps[idx]
        pred_temp = pred_temps[idx]
        
        # 计算误差
        error = np.abs(pred_temp - true_temp)
        mae = error.mean()
        max_error = error.max()
        
        # 真实值
        ax1 = plt.subplot(n_samples, 3, i*3 + 1)
        im1 = ax1.imshow(true_temp, cmap='hot', origin='lower')
        ax1.set_title(f'Ground Truth\n'
                     f'H={params[0]:.1f}mm, S={params[1]:.1f}mm\n'
                     f'k={params[2]:.0f}W/m·K, P={params[3]:.0f}W\n'
                     f'Max: {true_temp.max():.1f}°C',
                     fontsize=10, fontweight='bold')
        ax1.set_xlabel('x (mm)')
        ax1.set_ylabel('y (mm)')
        plt.colorbar(im1, ax=ax1, label='Temp (°C)')
        
        # 预测值
        ax2 = plt.subplot(n_samples, 3, i*3 + 2)
        im2 = ax2.imshow(pred_temp, cmap='hot', origin='lower', 
                        vmin=true_temp.min(), vmax=true_temp.max())  # 统一colorbar范围
        ax2.set_title(f'Prediction\n'
                     f'Max: {pred_temp.max():.1f}°C\n'
                     f'MAE: {mae:.2f}°C',
                     fontsize=10, fontweight='bold')
        ax2.set_xlabel('x (mm)')
        ax2.set_ylabel('y (mm)')
        plt.colorbar(im2, ax=ax2, label='Temp (°C)')
        
        # 误差图
        ax3 = plt.subplot(n_samples, 3, i*3 + 3)
        im3 = ax3.imshow(error, cmap='Reds', origin='lower')
        ax3.set_title(f'Absolute Error\n'
                     f'Mean: {mae:.2f}°C\n'
                     f'Max: {max_error:.2f}°C',
                     fontsize=10, fontweight='bold')
        ax3.set_xlabel('x (mm)')
        ax3.set_ylabel('y (mm)')
        plt.colorbar(im3, ax=ax3, label='Error (°C)')
    
    plt.suptitle('Model Performance: Prediction vs Ground Truth', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./outputs/prediction_vs_truth.png', 
                dpi=300, bbox_inches='tight')
    print('✓ 预测对比图已保存: prediction_vs_truth.png')
    plt.close()


# ============================================================================
# 5. 主程序
# ============================================================================

def main():
    """主函数（GPU优化版）"""
    
    print("\n" + "="*70)
    print("神经算子工程应用：散热器设计优化 (GPU加速版)")
    print("="*70)
    print("\n关键理解：")
    print("  × 不是：初始条件 → 时间演化")
    print("  ✓ 而是：设计参数 → 性能场")
    print("\n这才是工业界最常用的模式！\n")
    
    # 1. 设置GPU设备
    device = setup_device()
    
    # 参数
    n_train = 400
    n_test = 100
    grid_size = 64
    batch_size = 32  # GPU可以用更大的batch
    
    # 2. 生成训练数据（直接加载到GPU）
    print("\n【步骤1】生成训练数据")
    print("-"*70)
    train_params, train_temps = generate_heatsink_data(n_train, grid_size, device)
    
    print("\n【步骤2】生成测试数据")
    print("-"*70)
    test_params, test_temps = generate_heatsink_data(n_test, grid_size, device)
    
    # 3. 创建模型并移动到GPU
    print("\n【步骤3】创建模型")
    print("-"*70)
    model = TrueFNO(param_dim=4, grid_size=grid_size, width=32, modes=12).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"模型已加载到: {device}")
    
    # 4. GPU加速训练
    print("\n【步骤4】GPU加速训练")
    print("-"*70)
    model, param_mean, param_std = train_design_model_gpu(
        model, train_params, train_temps, device, 
        epochs=500, batch_size=batch_size
    )
    
    # 5. 测试性能
    print("\n【步骤5】测试性能")
    print("-"*70)
    model.eval()
    with torch.no_grad():
        test_params_norm = (test_params - param_mean) / param_std
        pred_temps = model(test_params_norm)
    
    # 计算误差
    mae = torch.abs(pred_temps - test_temps).mean().item()
    max_temp_error = torch.abs(
        pred_temps.max(dim=-1)[0].max(dim=-1)[0] - 
        test_temps.max(dim=-1)[0].max(dim=-1)[0]
    ).mean().item()
    
    print(f"平均绝对误差: {mae:.3f}°C")
    print(f"最高温度误差: {max_temp_error:.3f}°C")
    
    # ===== 新增：可视化预测vs真值 =====
    print("\n生成预测vs真值对比图...")
    visualize_predictions_vs_truth(
        test_params.cpu().numpy(), 
        test_temps.cpu().numpy(), 
        pred_temps.cpu().numpy()
    )
    
    # 6. GPU加速设计优化
    print("\n【步骤6】GPU加速设计优化")
    print("-"*70)
    best_params, best_temp = design_optimization_demo_gpu(
        model, param_mean, param_std, device, grid_size
    )
    
    # 总结
    print("\n" + "="*70)
    print("GPU加速总结")
    print("="*70)
    print("\nGPU优势：")
    print("  ✓ FFT运算：GPU比CPU快10-100倍")
    print("  ✓ 矩阵乘法：GPU并行计算")
    print("  ✓ 批处理：一次评估10000+设计")
    print("  ✓ 混合精度：FP16加速2-3倍")
    print("\n实际应用：")
    print("  - CPU训练：2-4小时/500 epochs")
    print("  - GPU训练：10-30分钟/500 epochs")
    print("  - 设计评估：10000个设计 < 1秒")
    print("  - 实时反馈：设计师可以交互式优化")
    print("\n这就是为什么GPU对神经算子如此重要！")
    print("="*70)


if __name__ == '__main__':
    main()