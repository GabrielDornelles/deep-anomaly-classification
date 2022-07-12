import torch
import torch.optim as optim
from training_engine import init_network_weights_from_pretraining, train_ae, train_encoder
from networks.models import DeepAutoEncoder, Encoder
import torchvision.transforms as transforms
from torchvision import datasets


device = torch.device("cuda")

# Dataset
#root = 'deep-anomaly-classification/samples' # a folder named with your class name and its images must be here
root = 'dataset'
size = 128
normal_class = 0

data_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])
 

dataset = datasets.ImageFolder(root, data_transforms)
        
train_set = dataset


train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
class_name = dataset.classes[normal_class]

# Train AutoEncoder
print("Starting AutoEncoder Training")
ae_net = DeepAutoEncoder()

ae_net.to(device)
ae_optimizer = optim.Adam(ae_net.parameters(), lr=3e-4, weight_decay=1e-6)
ae_net = train_ae(ae_net, ae_optimizer, train_loader, device, epoches=150)

torch.save(ae_net.state_dict(), "AutoEncoder.pth")

# Init Encoder with autoencoder weights
net = Encoder()
net = init_network_weights_from_pretraining(net, ae_net)
net.to(device)

# Train Net
lr_milestones = ()
optimizer = optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

model_dict = train_encoder(net=net, 
        train_loader=train_loader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        normal_class=normal_class, 
        lr_milestones=lr_milestones, 
        n_epochs=150, 
        device=device,
        warm_up_n_epochs=35,
        objective="soft-boundary")

torch.save(model_dict, f"DeepSVD_{class_name}.tar")