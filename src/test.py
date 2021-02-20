import torch
from tqdm import tqdm
from lib.metric_monitor import MetricMonitor
from lib.metrics import accuracy


def test(model, params, test_dataset_loader, criterion):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(test_dataset_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True).float()
            target = target.to(params["device"], non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            accuracy_result = accuracy(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy_result)
            stream.set_description(f"Test. {metric_monitor}")
