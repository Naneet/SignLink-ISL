import torch
from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self,model,device,optimizer,loss_fn, scheduler,train_path_list,test_path_list,save=False):
        self.device = device
        self.scaler = GradScaler(device)  # Initialize GradScaler
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_path_list = train_path_list
        self.test_path_list = test_path_list
        self.model = model
        self.save = save

    def train_step_pickle(self,epoch):
        self.model.train()
        train_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0
        

        for path in self.train_path_list:
            tensor = read_pickle(path=path)
            X = tensor['video'].to(torch.float16)
            y = tensor['label'].to(self.device)

             # Slice X into 4 parts
            if X.size(0) == 16:
                X_parts = torch.chunk(X, 4, dim=0)
                y_parts = torch.chunk(y, 4, dim=0)
            else:
                X_parts = [X]
                y_parts = [y]

            for X_part, y_part in zip(X_parts, y_parts):
                # Automatic Mixed Precision (AMP)
                with autocast(device_type=device):
                    y_pred = self.model(X_part.to(self.device))
                    loss = self.loss_fn(y_pred, y_part)
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
                total_correct += (y_pred_class == y_part).sum().item()
                total_samples += y_part.size(0)

                del X_part
                del y_part

        acc = total_correct * 100 / total_samples
        train_loss = train_loss / len(self.train_path_list)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Accuracy: {acc:.2f}")

            # Step scheduler if it's not ReduceLROnPlateau
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()


    def test_step_pickle(self, epoch):
        self.model.eval()
        test_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for path in self.test_path_list:
                tensor = read_pickle(path=path)
                X = tensor['video'].to(torch.float16)
                y = tensor['label'].to(self.device)

                # Slice X into 4 parts
                if X.size(0) == 16:
                    X_parts = torch.chunk(X, 4, dim=0)
                    y_parts = torch.chunk(y, 4, dim=0)
                else:
                    X_parts = [X]
                    y_parts = [y]

                for X_part, y_part in zip(X_parts, y_parts):
                    # Use AMP for inference
                    with autocast('cuda'):
                        y_pred = self.model(X_part.to(self.device))
                        loss = self.loss_fn(y_pred, y_part)
                        test_loss += loss.item()

                    # Compute predictions and accuracy
                    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                    total_correct += (y_pred_class == y_part).sum().item()
                    total_samples += y_part.size(0)

                    del X_part
                    del y_part

            acc = total_correct * 100 / total_samples
            test_loss = test_loss / len(self.test_path_list)
            print(f"Epoch: {epoch} | Test Loss: {test_loss:.4f} | Accuracy: {acc:.2f}")
            print("************************")

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
                }, filename=f"{self.model.name}_model={acc:.2f}_{epoch=}_real.pth")