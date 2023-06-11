import torch

from src.model import get_model
from src.loss import MaskedMSELoss, MaskedCrossEntropyLoss
from src.dataloader import get_data
from src.checkpoints import get_checkpoint_path

from config.utils import test_logger


def test(logging_path, config):

    # Get data
    test_data = get_data(config, 'test')
    test_num_users = test_data.get_num_user()
    test_ids, test_edge_index = test_data.get_input()
    test_target = test_data.get_target()

    # Get model and load model's weight
    model = get_model(config)
    checkpoint_path = get_checkpoint_path(config, logging_path)
    model.load_state_dict(torch.load(checkpoint_path))

    loss_and_metrics = []
    loss_and_metrics_value = []

    # Loss
    if config.model.loss == 'MaskedMSELoss':
        criterion = MaskedMSELoss()
        loss_and_metrics.append('MaskedMSELoss')

    elif config.model.loss == 'MaskedCrossEntropyLoss':
        criterion = MaskedCrossEntropyLoss()
        metric = MaskedMSELoss()
        loss_and_metrics.append('MaskedCrossEntropyLoss')
        loss_and_metrics.append('MaskedMSELoss')

    else:
        raise 'Error: loss must be MaskedMSELoss or MaskedCrossEntropyLoss but found ' + str(config.model.loss)

    # Final evaluation on the test's dataset
    model.eval()

    with torch.no_grad():
        model.eval()
        test_predict = model(test_ids, test_edge_index, test_num_users)
        loss = criterion(target=test_target, predict=test_predict)
        loss_and_metrics_value.append(loss.item())
        if config.model.loss == 'MaskedCrossEntropyLoss':
            test_metric = metric(target=test_target, predict=torch.argmax(test_predict, dim=2))
            loss_and_metrics_value.append(test_metric.item())

    print('test loss:', loss_and_metrics_value)
    test_logger(logging_path, loss_and_metrics, loss_and_metrics_value)

