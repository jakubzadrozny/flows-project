import torch
from torch import nn

def print_with_lines(msg):
    print()
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))
    print()

def train_vae(model, optim, train_loader, test_loader, epochs):
    hist = {
        "train_loss": [],
        "train_rec_loss": [],
        "train_kl": [],
        "test_loss": [],
        "test_rec_loss": [],
        "test_kl": [],
    }
        
    for epoch in range(1, epochs+1):
        print_with_lines(f"EPOCH {epoch}/{epochs}")
        
        model.train()
        
        epoch_loss = 0.
        epoch_rec_loss = 0.
        epoch_kl = 0.
        
        for batch_num, (x, _) in enumerate(train_loader, 1):
            rec, mu, log_var = model(x)
            
            losses = model.loss_function(x, rec, mu, log_var)
            loss = losses["loss"]
            
            epoch_loss += loss.item()
            epoch_rec_loss += losses["rec_loss"].item()
            epoch_kl += losses["kl"].item()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if batch_num % 25 == 0:
                print("[{}/{}] loss {:.3f}, rec-loss {:.3f}, kl {:.3f}".format(
                    batch_num,
                    len(train_loader.dataset) // train_loader.batch_size,
                    loss.item(),
                    losses["rec_loss"].item(),
                    losses["kl"].item(),
                ))
        
        mean_loss = epoch_loss / batch_num
        mean_rec_loss = epoch_rec_loss / batch_num
        mean_kl = epoch_kl / batch_num
        
        hist["train_loss"].append(mean_loss)
        hist["train_rec_loss"].append(mean_rec_loss)
        hist["train_kl"].append(mean_kl)
        
        print()
        print("===> Average train loss: {:.3f}".format(mean_loss))
        print("===> Average train rec-loss: {:.3f}".format(mean_rec_loss))
        print("===> Average train kl: {:.3f}".format(mean_kl))
        
        model.eval()
        
        epoch_loss = 0.
        epoch_rec_loss = 0.
        epoch_kl = 0.
        
        with torch.no_grad():
            for batch_num, (x, _) in enumerate(test_loader, 1):
                rec, mu, log_var = model(x)

                losses = model.loss_function(x, rec, mu, log_var)
                loss = losses["loss"]

                epoch_loss += loss.item()
                epoch_rec_loss += losses["rec_loss"].item()
                epoch_kl += losses["kl"].item()
        
        mean_loss = epoch_loss / batch_num
        mean_rec_loss = epoch_rec_loss / batch_num
        mean_kl = epoch_kl / batch_num
        
        hist["test_loss"].append(mean_loss)
        hist["test_rec_loss"].append(mean_rec_loss)
        hist["test_kl"].append(mean_kl)
        
        print()
        print("===> Average test loss: {:.3f}".format(mean_loss))
        print("===> Average test rec-loss: {:.3f}".format(mean_rec_loss))
        print("===> Average test kl: {:.3f}".format(mean_kl))
        
        # Save best model and a few samples per epoch
        
    print_with_lines("TRAINING FINISHED")
    
    for k, v in hist:
        hist[k] = torch.tensor(v)
        
    return hist