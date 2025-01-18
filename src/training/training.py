"""
Training script.
"""

import torch
#import torch.nn as nn
#import torch.nn.functional as F

def train_model(model, train_loader, val_loader, training_steps, criterion, optimizer, scheduler, device):
    """
    Train the NHITS model for a specified number of training steps.
    """
    model.to(device)
    
    # Check if the validation DataLoader is empty
    if len(val_loader.dataset) == 0:
        raise ValueError("The validation DataLoader is empty!")
        
    step = 0

    while step < training_steps:
        # Training loop
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            if step >= training_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        # Average the losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
         # Log the losses and learning rate
        print(f"Step {step}/{training_steps}, Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}, "
              f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Step the scheduler to decay the learning rate
        scheduler.step()
