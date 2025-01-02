import torch
from torch.amp import autocast, GradScaler
from utils.save_load_model import save_checkpoint
from utils.pickle import read_pickle

class Trainer:
    def __init__(self, model, device, optimizer, loss_fn, save=False):
        self.device = device
        self.scaler = GradScaler()  # Automatic Mixed Precision
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.save = save

    def train_step_pickle(self, epoch, preloaded_data):
        self.model.train()
        train_loss, total_correct, total_samples = 0, 0, 0

        for X, y in preloaded_data:
            # Forward pass with AMP
            with autocast('cuda'):
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss.item()

            # Backward pass with scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Accuracy computation
            y_pred_class = torch.argmax(y_pred, dim=1)
            total_correct += (y_pred_class == y).sum().item()
            total_samples += y.size(0)

        acc = total_correct * 100 / total_samples
        train_loss /= len(preloaded_data)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Accuracy: {acc:.2f}")

    def test_step_pickle(self, epoch,preloaded_data):
        self.model.eval()
        test_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in preloaded_data:
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
            test_loss = test_loss / len(preloaded_data)
            print(f"Epoch: {epoch} | Test Loss: {test_loss:.4f} | Accuracy: {acc:.2f}")

            if self.save:
                print(f"Saving model at epoch {epoch}...")
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_metric': acc
                }, filename=f"{self.model.name}_acc={acc:.2f}_{epoch=}_real.pth")

            print("************************")
