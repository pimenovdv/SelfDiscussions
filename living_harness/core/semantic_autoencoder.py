import torch
import torch.nn as nn
import torch.optim as optim
import os

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=64):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SemanticAutoencoderECC:
    def __init__(self, input_dim=256, hidden_dim=64, model_path="data/autoencoder.pth"):
        self.model = SimpleAutoencoder(input_dim, hidden_dim)
        self.model_path = model_path
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))

    def train_step(self, clean_vectors, noisy_vectors):
        """
        Denoising autoencoder training step.
        """
        self.model.train()
        clean_tensor = torch.tensor(clean_vectors, dtype=torch.float32)
        noisy_tensor = torch.tensor(noisy_vectors, dtype=torch.float32)

        self.optimizer.zero_grad()
        outputs = self.model(noisy_tensor)
        loss = self.criterion(outputs, clean_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def detect_and_correct(self, noisy_vector):
        """
        Corrects semantic shift using the autoencoder.
        """
        self.model.eval()
        with torch.no_grad():
            noisy_tensor = torch.tensor(noisy_vector, dtype=torch.float32).unsqueeze(0)
            corrected_tensor = self.model(noisy_tensor)
            return corrected_tensor.squeeze(0).tolist()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
