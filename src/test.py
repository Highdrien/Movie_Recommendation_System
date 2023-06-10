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

    # Loss
    if config.model.loss == 'MaskedMSELoss':
        criterion = MaskedMSELoss()
    elif config.model.loss == 'MaskedCrossEntropyLoss':
        criterion = MaskedCrossEntropyLoss()
    else:
        raise 'Error: loss must be MaskedMSELoss or MaskedCrossEntropyLoss but found ' + str(config.model.loss)

    # Final evaluation on the test's dataset
    model.eval()

    with torch.no_grad():
        model.eval()
        test_predict = model(test_ids, test_edge_index, test_num_users)
        loss = criterion(target=test_target, predict=test_predict)
        test_loss = loss.item()

    print('test loss:', test_loss)
    test_logger(logging_path, [config.model.loss], [test_loss])

