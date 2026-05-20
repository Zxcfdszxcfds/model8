import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ---------------------- 配置与初始化 ----------------------
st.set_page_config(page_title="Vibe Coding 生成模型实验", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载（使用Fashion-MNIST，无外网依赖）
@st.cache_resource
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = load_data()

# ---------------------- 1. Autoencoder 定义 ----------------------
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# ---------------------- 2. 简化VAE定义 ----------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

# ---------------------- 3. DCGAN定义 ----------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(z_dim, 256*7*7),
            nn.BatchNorm1d(256*7*7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256*7*7, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.main(x)

# ---------------------- 训练函数 ----------------------
def train_autoencoder(model, loader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in tqdm(loader):
            data = data.to(device).view(-1, 28*28)
            optimizer.zero_grad()
            recon, _ = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(loader.dataset)
        loss_history.append(avg_loss)
        print(f"AE Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return loss_history

def train_vae(model, loader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in tqdm(loader):
            data = data.to(device).view(-1, 28*28)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(data)
            recon_loss = nn.MSELoss(reduction='sum')(recon, data) / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(loader.dataset)
        loss_history.append(avg_loss)
        print(f"VAE Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return loss_history

def train_dcgan(netG, netD, loader, epochs=5, z_dim=100):
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    netG.train()
    netD.train()
    lossG_history = []
    lossD_history = []
    for epoch in range(epochs):
        total_lossG = 0
        total_lossD = 0
        for real, _ in tqdm(loader):
            real = real.to(device)
            b_size = real.size(0)
            label_real = torch.full((b_size, 1), 1.0, device=device)
            label_fake = torch.full((b_size, 1), 0.0, device=device)
            
            # 训练判别器
            netD.zero_grad()
            output_real = netD(real)
            loss_real = criterion(output_real, label_real)
            
            noise = torch.randn(b_size, z_dim, device=device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            loss_fake = criterion(output_fake, label_fake)
            
            lossD = loss_real + loss_fake
            lossD.backward()
            optimizerD.step()
            
            # 训练生成器
            netG.zero_grad()
            output = netD(fake)
            lossG = criterion(output, label_real)
            lossG.backward()
            optimizerG.step()
            
            total_lossG += lossG.item() * b_size
            total_lossD += lossD.item() * b_size
        
        avg_lossG = total_lossG / len(loader.dataset)
        avg_lossD = total_lossD / len(loader.dataset)
        lossG_history.append(avg_lossG)
        lossD_history.append(avg_lossD)
        print(f"DCGAN Epoch {epoch+1}, LossG: {avg_lossG:.4f}, LossD: {avg_lossD:.4f}")
    return lossG_history, lossD_history

# ---------------------- Streamlit界面 ----------------------
st.title("🎨 Vibe Coding 生成模型实验平台")
tab1, tab2, tab3 = st.tabs([
    "1. Autoencoder vs VAE 重构对比",
    "2. VAE 隐空间可视化与插值",
    "3. DCGAN 生成实验"
])

# ---------------------- 模块1：Autoencoder vs VAE ----------------------
with tab1:
    st.header("Autoencoder 与 VAE 重构对比")
    latent_dim = st.slider("隐空间维度", 8, 64, 32)
    epochs = st.slider("训练轮数", 1, 10, 3)
    
    if st.button("开始训练与对比", key="train_ae_vae"):
        with st.spinner("训练中..."):
            # 初始化模型
            ae = Autoencoder(latent_dim).to(device)
            vae = VAE(latent_dim).to(device)
            
            # 训练
            ae_loss = train_autoencoder(ae, train_loader, epochs=epochs)
            vae_loss = train_vae(vae, train_loader, epochs=epochs)
            
            # 测试集重构
            ae.eval()
            vae.eval()
            with torch.no_grad():
                test_imgs, labels = next(iter(test_loader))
                test_imgs = test_imgs.to(device)
                ae_recon, _ = ae(test_imgs)
                vae_recon, _, _, _ = vae(test_imgs)
            
            # 可视化结果
            fig, axes = plt.subplots(3, 5, figsize=(12, 6))
            for i in range(5):
                axes[0,i].imshow(test_imgs[i].view(28,28).cpu().numpy()*0.5+0.5, cmap='gray')
                axes[0,i].set_title("Original")
                axes[0,i].axis('off')
                axes[1,i].imshow(ae_recon[i].view(28,28).cpu().numpy()*0.5+0.5, cmap='gray')
                axes[1,i].set_title("AE Recon")
                axes[1,i].axis('off')
                axes[2,i].imshow(vae_recon[i].view(28,28).cpu().numpy()*0.5+0.5, cmap='gray')
                axes[2,i].set_title("VAE Recon")
                axes[2,i].axis('off')
            st.pyplot(fig)
            
            # 损失曲线
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
            ax1.plot(ae_loss, label='Autoencoder Loss')
            ax1.set_title('Autoencoder Loss Curve')
            ax1.legend()
            ax2.plot(vae_loss, label='VAE Loss')
            ax2.set_title('VAE Loss Curve')
            ax2.legend()
            st.pyplot(fig2)

# ---------------------- 模块2：VAE隐空间可视化与插值 ----------------------
with tab2:
    st.header("VAE 隐空间可视化与插值")
    if st.button("加载训练好的VAE并可视化", key="vae_vis"):
        with st.spinner("处理中..."):
            # 加载模型（这里直接用上面训练的，也可以加载本地模型）
            vae = VAE(latent_dim=32).to(device)
            # 训练（为了演示，这里简化，实际可加载预训练模型）
            vae_loss = train_vae(vae, train_loader, epochs=3)
            vae.eval()
            
            # 隐空间散点图
            z_list = []
            label_list = []
            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(device).view(-1, 28*28)
                    _, mu, _, z = vae(data)
                    z_list.append(mu.cpu().numpy())
                    label_list.append(labels.numpy())
            z_all = np.concatenate(z_list, axis=0)
            labels_all = np.concatenate(label_list, axis=0)
            
            # PCA降维到2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_all)
            
            fig, ax = plt.subplots(figsize=(8,6))
            scatter = ax.scatter(z_2d[:,0], z_2d[:,1], c=labels_all, cmap='tab10', s=5)
            plt.colorbar(scatter, ticks=range(10), label='Fashion-MNIST Class')
            st.pyplot(fig)
            
            # 隐空间插值
            idx1, idx2 = 0, 100
            img1, img2 = test_loader.dataset[idx1][0].unsqueeze(0).to(device), test_loader.dataset[idx2][0].unsqueeze(0).to(device)
            with torch.no_grad():
                _, mu1, _, _ = vae(img1.view(-1,28*28))
                _, mu2, _, _ = vae(img2.view(-1,28*28))
            
            steps = 10
            alphas = np.linspace(0,1,steps)
            interpolated_imgs = []
            for alpha in alphas:
                z_interp = (1-alpha)*mu1 + alpha*mu2
                recon = vae.decoder(z_interp).view(28,28).cpu().numpy()*0.5+0.5
                interpolated_imgs.append(recon)
            
            fig, axes = plt.subplots(1, steps, figsize=(15,2))
            for i, img in enumerate(interpolated_imgs):
                axes[i].imshow(img, cmap='gray')
                axes[i].axis('off')
            st.pyplot(fig)

# ---------------------- 模块3：DCGAN生成实验 ----------------------
with tab3:
    st.header("DCGAN 生成实验")
    z_dim = 100
    epochs_gan = st.slider("DCGAN训练轮数", 1, 10, 3)
    seed = st.number_input("随机种子", value=42)
    torch.manual_seed(seed)
    
    if st.button("训练DCGAN并生成", key="train_gan"):
        with st.spinner("训练中..."):
            netG = Generator(z_dim).to(device)
            netD = Discriminator().to(device)
            lossG, lossD = train_dcgan(netG, netD, train_loader, epochs=epochs_gan, z_dim=z_dim)
            
            # 生成不同噪声的样本
            noise = torch.randn(8, z_dim, device=device)
            with torch.no_grad():
                fake_imgs = netG(noise).view(-1,1,28,28).cpu()*0.5+0.5
            
            fig, axes = plt.subplots(2,4, figsize=(10,5))
            for i in range(8):
                axes[i//4, i%4].imshow(fake_imgs[i,0], cmap='gray')
                axes[i//4, i%4].axis('off')
            st.pyplot(fig)
            
            # 损失曲线
            fig2, ax = plt.subplots(figsize=(8,4))
            ax.plot(lossG, label='Generator Loss')
            ax.plot(lossD, label='Discriminator Loss')
            ax.legend()
            st.pyplot(fig2)

# ---------------------- 页脚 ----------------------
st.markdown("---")
st.caption("模式识别与图像处理 - A8 Vibe Coding 实验")
