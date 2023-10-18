import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import time
import datetime
import copy


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def init_network_weights_from_pretraining(net, ae_net):
    """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

    net_dict = net.state_dict()
    ae_net_dict = ae_net.state_dict()

    # Filter out decoder network keys
    ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
    # Overwrite values in the existing state_dict
    net_dict.update(ae_net_dict)
    # Load the new state_dict
    net.load_state_dict(net_dict)
    return net

def init_center_c(train_loader: DataLoader, net: object, eps=0.1, device=torch.device("cuda")):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def plot_losses(losses, figname):
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(losses, color='blue', label='Training loss') 
    ax.set(title="Loss ", 
            xlabel='Steps',
            ylabel='Loss') 
    ax.legend()
    plt.savefig(figname)

def train_ae(model, optimizer, train_loader, device = torch.device("cuda"), epoches=150):
    losses = []
    model.train()
    lowest_loss = 1e6
    for epoch in range(epoches):
        
        loss_epoch = 0.0
        n_batches = 0
        
        for data in train_loader:
            x, _ = data
            x = x.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            scores = torch.sum((outputs - x) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            losses.append(loss)
            n_batches +=1

        epoch_loss = loss_epoch/n_batches
        print(f"Epoch {epoch+1}/{epoches}           Loss: {epoch_loss:.3f}")
        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())

    
    losses = [item.detach().cpu().numpy() for item in losses]
    plot_losses(losses, "results/autoencoder_loss.png")
    model.load_state_dict(best_weights)
    return model


def train_encoder(net, train_loader, optimizer, scheduler, lr_milestones, normal_class, 
                warm_up_n_epochs=35, n_epochs=150, device=torch.device("cuda"), 
                objective="soft-boundary"):

    nu = 0.1
    R = 0.0
    R = torch.tensor(R, device=device)  # radius R initialized with 0 by default.
    c = None
    losses = []
   
    if c is None:
        print('Initializing center c.')
        c = init_center_c(train_loader, net)
        print('Center c initialized.')

    start_time = time.time()
    net.train()
    lowest_loss = 1e6
    print("Starting autoencoder Training")
    for epoch in range(n_epochs):

        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            # Zero the network parameter gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = net(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            if objective == 'soft-boundary':
                scores = dist - R ** 2
                loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
            else:
                loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            scheduler.step()
            if epoch in lr_milestones:
                print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            # Update hypersphere radius R on mini-batch distances
            if (objective == 'soft-boundary') and (epoch >= warm_up_n_epochs):
                R.data = torch.tensor(get_radius(dist, nu), device=device)

            loss_epoch += loss.item()
            losses.append(loss)
            n_batches += 1
        
        check_loss = loss_epoch / n_batches
        if check_loss < lowest_loss:
            lowest_loss = check_loss
            print(f"Lower loss found: {lowest_loss}")
            best_weights = copy.deepcopy(net.state_dict())

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        print('  Epoch {}/{}\t Batch inference Time: {:.3f}\t Loss: {:.3f}'
                    .format(epoch + 1, n_epochs, epoch_train_time, loss_epoch / n_batches))

    train_time = time.time() - start_time
    print(f"Total Training time: {str(datetime.timedelta(seconds=train_time))}")
    losses = [item.detach().cpu().numpy() for item in losses]
    plot_losses(losses, "results/encoder_loss.png")
    net.load_state_dict(best_weights)
    model_dict = {
        "radius": R,
        "center": c,
        "encoder_weights": net.state_dict(),
        "trained_class_idx": normal_class 
    }
    
    return model_dict
