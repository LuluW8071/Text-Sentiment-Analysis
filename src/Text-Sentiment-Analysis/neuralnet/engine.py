import torch 
import numpy as np
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torchmetrics import Accuracy


device = "cuda" if torch.cuda.is_available() else "cpu"
torchmetrics_accuracy = Accuracy(task = "multiclass", num_classes = 3).to(device)

def train_step(model, data_loader, loss_fn, optimizer, clip, device):  
    train_loss, train_acc = 0, 0
    train_losses, train_accuracies = [], []

    model.to(device)
    model.train()

    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        output, _ = model(inputs)

        loss = loss_fn(output, labels.long())
        train_loss += loss

        preds = torch.argmax(output, dim=1)
        train_acc += torchmetrics_accuracy(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    train_losses.append(train_loss.item())
    train_accuracies.append(train_acc.item())

    return train_losses, train_accuracies

def test_step(model, data_loader, loss_fn, device, classes, experiment):
    test_loss, test_acc = 0, 0
    test_losses, test_accuracies = [], []
    preds_list, labels_list = [], []

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            output, _ = model(inputs)

            loss = loss_fn(output, labels.long())
            test_loss += loss

            preds = torch.argmax(output, dim=1)
            test_acc += torchmetrics_accuracy(preds, labels)

            # Collect predictions and labels for confusion matrix
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        test_losses.append(test_loss.item())
        test_accuracies.append(test_acc.item())

        # Log the confusion matrix to Comet
        experiment.log_confusion_matrix(
            y_true=np.array(labels_list),
            y_predicted=np.array(preds_list),
            labels=classes,
        )
        return test_losses, test_accuracies
    
def train(model, train_dataloader, test_dataloader, classes, loss_fn, optimizer, clip, epochs, device, experiment):
    results = {"train_losses_history": [],
               "train_accuracies_history": [],
               "test_losses_history": [],
               "test_accuracies_history": []

    }
    test_acc_min = 0

    for epoch in tqdm(range(1, epochs+1)):
        train_losses, train_accuracies = train_step(model,
                                                    train_dataloader,
                                                    loss_fn, optimizer, clip,
                                                    device)
        test_losses, test_accuracies = test_step(model,
                                                 test_dataloader,
                                                 loss_fn,
                                                 device, 
                                                 classes,
                                                 experiment)

        # Calculate avg. loss and accuracy
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_accuracy = sum(test_accuracies) / len(test_accuracies)

        print(f'\nTrain loss: {avg_train_loss:.4f} ---- Train acc: {avg_train_accuracy:.4f}')
        print(f'test loss: {avg_test_loss:.4f} ---- test acc: {avg_test_accuracy:.4f}\n')

        # Save model if avg_test_accuracy is higher
        if avg_test_accuracy > test_acc_min:
            torch.save(model.state_dict(), 'best_model.pth')
            test_acc_min = avg_test_accuracy
            print(f"Saved best model at epoch: {epoch}\n")

            # Log metrics to a dictionary
            metrics_dict = {
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_accuracy,
                "test_loss": avg_test_loss,
                "test_accuracy": avg_test_accuracy,
            }

            # Log metrics dictionary to Comet ML
            experiment.log_metrics(metrics_dict, step=epoch)

        results["train_losses_history"].append(avg_train_loss)
        results["train_accuracies_history"].append(avg_train_accuracy)
        results["test_losses_history"].append(avg_test_loss)
        results["test_accuracies_history"].append(avg_test_accuracy)

    return results