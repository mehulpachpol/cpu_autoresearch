import torch
import torch.nn as nn
import torch.optim as optim
import time
from prepare import get_dataloaders, evaluate_accuracy

# Baseline Configuration (The AI should modify this)
HIDDEN_SIZE = 32
LEARNING_RATE = 0.01

# Baseline Model: A simple 2-layer MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST images are 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, HIDDEN_SIZE) 
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10) # 10 classes (digits 0-9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    device = torch.device('cpu')
    model = SimpleMLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
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
                
            images = images.view(images.shape[0], -1).to(device)
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