import torch
import torch.nn as nn
import torch.optim as optim
import time
from prepare import get_dataloaders, evaluate_accuracy

# Hypothesis: a small CNN should extract MNIST structure far better than a tiny MLP
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, 28, 28)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

def train():
    device = torch.device('cpu')
    model = SimpleCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    trainloader, valloader = get_dataloaders()
    
    # The brilliant constraint: exactly 60 seconds of training
    START_TIME = time.time()
    TIME_BUDGET_SECONDS = 60 
    
    print("Starting 60-second training run...")
    model.train()
    
    step = 0
    # Infinite loop broken only by time
    while True:
        for images, labels in trainloader:
            if time.time() - START_TIME >= TIME_BUDGET_SECONDS:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            step += 1
            
        if time.time() - START_TIME >= TIME_BUDGET_SECONDS:
            break
            
    print(f"Time budget exhausted. Completed {step} steps.")
    
    # Run the immutable evaluation script
    final_accuracy = evaluate_accuracy(model, valloader)
    print(f"Final Validation Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    train()
