from tqdm import tqdm

import torch
import torch.nn as nn

from src.model import get_model
from src.dataloader import creates_generators


def test(config):

    _, _, test_loader = creates_generators(config)

    # Instancier le modèle
    model = get_model()

    # Load model's weight
    model.load_state_dict(torch.load(checkpoint_path))

    # Définir la loss
    criterion = nn.MSELoss()

    # Évaluation finale sur l'ensemble de test
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            user_ids = inputs[:, 0]
            item_ids = inputs[:, 1]

            outputs = model(user_ids, item_ids)
            loss = criterion(outputs.squeeze(), targets)

            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader)

    print(f"Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    checkpoint_path = 'model.pth'
    test(checkpoint_path)


