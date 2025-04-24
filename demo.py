import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from net import STgramMFN, TgramNet
from torchvision.transforms import ToTensor
# demo
# ------------------------------
# 1. 生成模拟音频数据（或加载真实音频）
# ------------------------------
def generate_demo_audio(duration=2.0, sample_rate=16000):
    """生成包含正弦波和噪声的模拟音频"""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    signal = 0.5 * torch.sin(2 * np.pi * 440 * t)  # 440Hz正弦波
    noise = 0.1 * torch.randn_like(t)              # 高斯噪声
    audio = signal + noise
    return audio, sample_rate

# 生成音频（形状: [1, 32000]）
audio, sr = generate_demo_audio()
print(f"原始音频形状: {audio.shape}, 采样率: {sr}")

# ------------------------------
# 2. 可视化原始波形
# ------------------------------
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(audio.numpy().squeeze(), color='blue')
plt.title("原始音频波形")
plt.xlabel("采样点")
plt.ylabel("振幅")

# ------------------------------
# 3. 提取梅尔频谱（模拟ASDDataset.transform）
# ------------------------------
def extract_mel(audio, sr=16000, n_mels=128, n_fft=1024, hop_len=512):
    """提取梅尔频谱（对数刻度）"""
    # STFT计算
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_len)(audio)
    # 梅尔滤波器
    mel_spec = torchaudio.transforms.MelScale(
        n_mels=n_mels, sample_rate=sr)(spec)
    # 对数压缩
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return log_mel

mel_spec = extract_mel(audio, sr)
print(f"梅尔频谱形状: {mel_spec.shape}")  # [1, 128, 63]

# 可视化梅尔频谱
plt.subplot(3, 1, 2)
plt.imshow(mel_spec.squeeze().numpy(),
           aspect='auto', origin='lower', cmap='viridis')
plt.title("梅尔频谱图")
plt.xlabel("时间帧")
plt.ylabel("梅尔频带")

# ------------------------------
# 4. 提取时频谱（TgramNet处理）
# ------------------------------
tgram_net = TgramNet(mel_bins=128, win_len=1024, hop_len=512)
tgram = tgram_net(audio.unsqueeze(0))  # 输入: [1, 1, 32000]
print(f"时频谱形状: {tgram.shape}")     # [1, 128, 63]

# 可视化时频谱
plt.subplot(3, 1, 3)
plt.imshow(tgram.squeeze().detach().numpy(),
           aspect='auto', origin='lower', cmap='viridis')
plt.title("时频谱图 (TgramNet输出)")
plt.xlabel("时间帧")
plt.ylabel("频带")
plt.tight_layout()
plt.show()

# ------------------------------
# 5. 模型完整处理流程（STgramMFN）
# ------------------------------
# 初始化模型（假设2分类：正常/异常）
model = STgramMFN(num_classes=2, use_arcface=False)
model.eval()

# 模拟输入（需与训练时预处理一致）
x_wav = audio.unsqueeze(0)            # [1, 32000]
x_mel = mel_spec.unsqueeze(0)         # [1, 1, 128, 63]
label = torch.tensor([0])             # 假设标签为"正常"

# 前向传播
with torch.no_grad():
    logits, feature = model(x_wav, x_mel, label)
    prob = torch.softmax(logits, dim=1)

print(f"模型输出logits: {logits}")
print(f"类别概率: 正常={prob[0][0]:.4f}, 异常={prob[0][1]:.4f}")

# ------------------------------
# 6. 异常分数计算（GMM模式可选）
# ------------------------------
if hasattr(model, 'arcface') and model.arcface:
    print("\n使用ArcFace特征空间距离作为异常分数:")
    normalized_feat = F.normalize(feature, p=2, dim=1)
    score = -torch.norm(normalized_feat - model.arcface.weight[0], dim=1)
    print(f"异常分数: {score.item():.4f} (越小越异常)")