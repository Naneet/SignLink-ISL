import torch
from torch.amp import autocast, GradScaler
from utils.save_load import save_checkpoint
from utils.pickle import read_pickle

class Trainer:
    def __init__(self,model,device,optimizer,scheduler,loss_fn,train_dataloader,test_dataloader,save=False):
        self.device = device
        self.scaler = GradScaler(device)  # Initialize GradScaler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.model = model
        self.save = save

    def train_step(self,epoch):
        self.model.train()
        train_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0

        for X, y in self.train_dataloader:
            X, y = X.to(self.device), y.to(self.device)

            # Automatic Mixed Precision (AMP)
            with autocast('cuda'):
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()

            # Scale the loss and call backward
            self.scaler.scale(loss).backward()
            # Update parameters with scaled gradients
            self.scaler.step(self.optimizer)
            # Update scaler for next iteration
            self.scaler.update()

            self.optimizer.zero_grad()

            # Compute predictions and accuracy
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            total_correct += (y_pred_class == y).sum().item()
            total_samples += y.size(0)

        acc = total_correct * 100 / total_samples
        train_loss = train_loss / len(self.train_dataloader)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Accuracy: {acc:.2f}")

        # Step scheduler if ReduceLROnPlateau
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_loss)  # Use validation loss as the metric


    def test_step(self, epoch):
        self.model.eval()
        test_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                # Use AMP for inference
                with autocast('cuda'):
                    y_pred = self.model(X)
                    loss = self.loss_fn(y_pred, y)
                    test_loss += loss.item()

                # Compute predictions and accuracy
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                total_correct += (y_pred_class == y).sum().item()
                total_samples += y.size(0)

            acc = total_correct * 100 / total_samples
            test_loss = test_loss / len(self.test_dataloader)
            print(f"Epoch: {epoch} | Test Loss: {test_loss:.4f} | Accuracy: {acc:.2f}")

            # Step scheduler if ReduceLROnPlateau
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(test_loss)  # Use validation loss as the metric
            
            if self.save:
                print(f"Saving model at epoch {epoch}...")
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': acc
                }, filename=f"/saved/{self.model.name}_model=acc_{acc:.2f}_loss_{test_loss}_{epoch=}_real.pth")

            print("************************")