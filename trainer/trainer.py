import torch
class Trainer:
    def __init__(self,device):
        self.device = device

    def train_step(self, model, optimizer, loss_fn, epoch, dataloader):
        model.train()
        train_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0

        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss +=loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            total_correct += (y_pred_class == y).sum().item()
            total_samples += y.size(0)

        acc = total_correct * 100 / total_samples

        train_loss = train_loss/len(dataloader)
        print(f"Epoch: {epoch} | Loss: {loss} | Accuracy: {acc:2f}")


    def test_step(self, model, loss_fn, epoch, dataloader):
        model.eval()
        test_loss, acc = 0, 0
        total_correct = 0
        total_samples = 0


        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                test_loss += loss.item()
                y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                total_correct += (y_pred_class == y).sum().item()
                total_samples += y.size(0)

            acc = total_correct * 100 / total_samples
            global best_metric
            test_loss = test_loss/len(dataloader)
            print(f"Epoch: {epoch} | Test Loss: {loss} | Accuracy: {acc:2f}")
            print("************************")