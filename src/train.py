from tqdm import tqdm
from lib.metric_monitor import MetricMonitor
from lib.metrics import accuracy


def train(model, params, train_dataset_loader, criterion, optimizer, epoch):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_dataset_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True).float()
        target = target.to(params["device"], non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        accuracy_result = accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy_result)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(f"Epoch: {epoch}. Train. {metric_monitor}")
