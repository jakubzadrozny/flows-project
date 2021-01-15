import torch
from torch import nn

def print_with_lines(msg):
    print()
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))
    print()

def train_real_nvp(model, optim, train_loader, test_loader, epochs, scale_reg=5e-5):
    image_size = datainfo.channel * datainfo.size ** 2
    
    train_loss_hist = []
    train_log_ll_hist = []
    train_bits_per_dim_hist = []
    test_loss_hist = []
    test_log_ll_hist = []
    test_bits_per_dim_hist = []
        
    for epoch in range(1, epochs+1):
        print_with_lines(f"EPOCH {epoch}/{epochs}")
        
        model.train()
        
        epoch_loss = 0.
        epoch_log_ll = 0.
        
        for batch_num, (x, _) in enumerate(train_loader, 1):
            x, log_det = logit_transform(x)
            
            log_ll, weight_scale = model(x)
            
            log_ll = (log_ll + log_det).mean()
            loss = -log_ll + scale_reg * weight_scale
            
            epoch_loss += loss.item()
            epoch_log_ll += log_ll.item()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if batch_num % 1 == 0:
                bits_per_dim = 8 -log_ll.item() / (image_size * np.log(2.))
                print("[{}/{}] loss {:.3f}, log-ll {:.3f}, bits/dim {:.3f}".format(
                    batch_num,
                    len(train_loader.dataset) // train_loader.batch_size,
                    loss.item(),
                    log_ll.item(),
                    bits_per_dim
                ))
        
        mean_loss = epoch_loss / batch_num
        mean_log_ll = epoch_log_ll / batch_num
        mean_bits_per_dim = 8 - mean_log_ll / (image_size * np.log(2.))
        
        train_loss_hist.append(mean_loss)
        train_log_ll_hist.append(mean_log_ll)
        train_bits_per_dim_hist.append(mean_bits_per_dim)
        
        print()
        print("===> Average train loss: {:.3f}".format(mean_loss))
        print("===> Average train log-likelihood: {:.3f}".format(mean_log_ll))
        print("===> Average train bits/dim: {:.3f}".format(mean_bits_per_dim))
        
        model.eval()
        
        epoch_loss = 0.
        epoch_log_ll = 0.
        
        with torch.no_grad():
            for batch_num, (x, _) in enumerate(test_loader, 1):
                x, log_det = logit_transform(x)
                log_ll, weight_scale = model(x)

                log_ll = (log_ll + log_det).mean()
                loss = -log_ll + scale_reg * weight_scale

                epoch_loss += loss.item()
                epoch_log_ll += log_ll.item()
        
        mean_loss = epoch_loss / batch_num
        mean_log_ll = epoch_log_ll / batch_num
        mean_bits_per_dim = 8 - mean_log_ll / (image_size * np.log(2.))
        
        test_loss_hist.append(mean_loss)
        test_log_ll_hist.append(mean_log_ll)
        test_bits_per_dim_hist.append(mean_bits_per_dim)
        
        print()
        print("===> Average test loss: {:.3f}".format(mean_loss))
        print("===> Average test log-likelihood: {:.3f}".format(mean_log_ll))
        print("===> Average test bits/dim: {:.3f}".format(mean_bits_per_dim))
        
        # Save best model and a few samples per epoch
        
    print_with_lines("TRAINING FINISHED")
    
    hist = {
        "train_loss": torch.tensor(train_loss_hist),
        "train_log_ll": torch.tensor(train_log_ll_hist),
        "train_bits_per_dim": torch.tensor(train_bits_per_dim),
        "test_loss": torch.tensor(test_loss_hist),
        "test_log_ll": torch.tensor(test_log_ll_hist),
        "test_bits_per_dim": torch.tensor(test_bits_per_dim),
    }
    
    return hist