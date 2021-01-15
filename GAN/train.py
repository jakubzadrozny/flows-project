import torch
from torch import nn

def print_with_lines(msg):
    print()
    print("-" * len(msg))
    print(msg)
    print("-" * len(msg))
    print()

def train_gan(G, D, optimG, optimD, loader, epochs, real_label=1, fake_label=0):
    hist = {
        "G_loss": [],
        "D_loss": [],
    }
    
    criterion = nn.BCELoss()
        
    for epoch in range(1, epochs+1):
        print_with_lines(f"EPOCH {epoch}/{epochs}")
        
        model.train()
        
        epoch_G_loss = 0.
        epoch_D_loss = 0.
        
        for batch_num, (x, _) in enumerate(loader, 1):
            ## Train with all-real batch
            D.zero_grad()
            
            # Format batch
            b_size = x.shape[0]
            label = torch.full((b_size,), real_label)
            
            # Forward pass real batch through D
            output = D(x)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, D.nz, 1, 1)
            
            # Generate fake image batch with G
            fake = G(noise)
            label = torch.full((b_size,), fake_label)
            
            # Classify all fake batch with D
            output = D(fake.detach())
            
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            epoch_D_loss += errD.item()
            
            # Update D
            optimizerD.step()
            
            ## Train G
            G.zero_grad()
            label = torch.full((b_size,), fake_label)
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = D(fake)

            # Calculate G's loss based on this output
            errG = criterion(output, label)
            epoch_G_loss += errG.item()
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Update G
            optimizerG.step()
            
            if batch_num % 25 == 0:
                print("[{}/{}] D-loss {:.3f}, G-loss {:.3f}".format(
                    batch_num,
                    len(train_loader.dataset) // train_loader.batch_size,
                    errD.item(),
                    errG.item()
                ))
        
        mean_D_loss = epoch_D_loss / batch_num
        mean_G_loss = epoch_G_loss / batch_num
        
        hist["D_loss"].append(mean_D_loss)
        hist["G_loss"].append(mean_G_loss)
        
        print()
        print("===> Average D-loss: {:.3f}".format(mean_D_loss))
        print("===> Average G-loss: {:.3f}".format(mean_G_loss))
        
        # Save best model and a few samples per epoch
        
    print_with_lines("TRAINING FINISHED")
    
    for k, v in hist:
        hist[k] = torch.tensor(v)
        
    return hist