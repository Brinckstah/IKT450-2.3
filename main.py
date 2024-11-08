import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
#from vgg_pytorch import VGG 

class ResNetModel(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True) # 34, 50, 101, 152 for flere convl
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(128, 11)  # 11 classes
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56) # flattening tensor to single dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # downsize 224 to go fasteeer
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ROOT_PATH = 'food11 dataset'
    training_path = f'{ROOT_PATH}/training'
    validation_path = f'{ROOT_PATH}/validation'
    evaluation_path = f'{ROOT_PATH}/evaluation'
    
    train_dataset = datasets.ImageFolder(training_path, transform=transform)
    evaluation_dataset = datasets.ImageFolder(evaluation_path, transform=transform)
    validation_dataset = datasets.ImageFolder(validation_path, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    evaluation_dataloader = DataLoader(evaluation_dataset, batch_size=32, shuffle=True, num_workers=2)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    return train_dataloader, evaluation_dataloader, validation_dataloader

def train_model(model, train_loader, criterion, optimizer, epoch, device):
    model.to(device)
    model.train() # Training for each epoch
    model.to(device)
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        #loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
            epoch, 100. * batch_idx / len(train_loader), loss.item()))

def test_model(model, val, criterion, device):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val.dataset),
        100. * correct / len(val.dataset)))
    return all_labels, all_preds
    
def results(all_labels, all_preds, label_names):
    #all_preds = [p[0] for p in all_preds]  # Flatten predictions
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=label_names, labels=range(11), digits=2)
    print("\nTest Classification Report:\n")
    print(report)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    DATASET_LABELS = {
    '0': "Bread",
    '1': "Dairy product",
    '2': "Dessert",
    '3': "Egg",
    '4': "Fried food",
    '5': "Meat",
    '6': "Noodles/Pasta",
    '7': "Rice",
    '8': "Seafood",
    '9': "Soup",
    '10': "Vegetable/Fruit"
    }
    label_names = [DATASET_LABELS[str(i)] for i in range(11)]

    train_dataloader, evaluation_dataloader, test_dataloader = dataloader()
    #model = SimpleCNN()
    model = ResNetModel()
    #model = VGG.from_pretrained('vgg11', num_classes=10)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for epoch in range(1, epochs + 1):
        train_model(model, train_dataloader, criterion, optimizer, epoch, device=device)
        all_labels, all_preds = test_model(model, evaluation_dataloader, criterion, device=device)
        print(f'Labels: {all_labels}')
        print(f'Labels: {all_preds}')
    results(all_labels, all_preds, label_names)

if __name__ == '__main__':
    main()