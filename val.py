import torch
from torchmetrics import MeanSquaredError, MeanAbsolutePercentageError, R2Score, MeanAbsoluteError
from torch.autograd import Variable
from torchvision import transforms
from dataloader import listDataset
from utils import prepare_datasets

from tqdm import tqdm


class Metrics:
    """
    A class to calculate and update evaluation metrics for model predictions.

    Metrics include:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R2 Score
    - Mean Absolute Percentage Error (MAPE)
    """

    def __init__(self, device):
        """
        Initialize the metrics class.

        Args:
            device (str): Device to perform calculations on ('cuda' or 'cpu').
        """
        self.device = device
        # Placeholder tensors for predictions and targets
        self.preds = torch.tensor(0, device=device)
        self.targets = torch.tensor(0, device=device)

        # Initialize metrics on the specified device
        self.__MAELoss = MeanAbsoluteError().to(device)
        self.__RMSELoss = MeanSquaredError(squared=False).to(device)
        self.__R2Loss = R2Score().to(device)
        self.__MAPELoss = MeanAbsolutePercentageError().to(device)

    def update(self, preds, targets):
        """
        Update the metrics with new predictions and targets.

        Args:
            preds (torch.Tensor): Model predictions of shape [B, C, H, W].
            targets (torch.Tensor): Ground truth of shape [B, H, W].

        Raises:
            AssertionError: If the shape of `preds` or `targets` is incorrect.
        """
        assert len(preds.shape) == 4, "Wrong shape. Predictions must be [B, C, H, W]"
        assert len(targets.shape) == 3, "Wrong shape. Targets must be [B, H, W]"

        # Process predictions: remove channel dimension and sum over spatial dimensions
        preds = torch.squeeze(preds, dim=1).sum(dim=(1, 2)).to(self.device)
        # Sum over spatial dimensions for targets
        targets = targets.sum(dim=(1, 2)).to(self.device)

        # Concatenate with existing predictions and targets
        if not self.preds.shape:  # Initialize tensors on the first update
            self.preds = preds
            self.targets = targets
        else:
            self.preds = torch.cat((self.preds, preds), 0)
            self.targets = torch.cat((self.targets, targets), 0)

    def RMSELoss(self):
        """
        Compute Root Mean Squared Error (RMSE).

        Returns:
            torch.Tensor: RMSE value.
        """
        return self.__RMSELoss(self.preds, self.targets)

    def R2Loss(self):
        """
        Compute R2 Score.

        Returns:
            torch.Tensor: R2 Score value.
        """
        return self.__R2Loss(self.preds, self.targets)

    def MAPELoss(self):
        """
        Compute Mean Absolute Percentage Error (MAPE).

        Returns:
            torch.Tensor: MAPE value.
        """
        return self.__MAPELoss(self.preds, self.targets)

    def MAELoss(self):
        """
        Compute Mean Absolute Error (MAE).

        Returns:
            torch.Tensor: MAE value.
        """
        return self.__MAELoss(self.preds, self.targets)


def validate(val_list, model, criterion, config):
    """
    Validate the model on a given validation dataset.

    Args:
        val_list (list): List of file paths for validation images.
        model (torch.nn.Module): Trained model to evaluate.
        criterion (callable): Loss function used for evaluation (e.g., MSELoss).
        config (dict): Configuration dictionary containing dataset name, batch size, etc.

    Returns:
        float: Mean Absolute Error (MAE) calculated over the validation dataset.
    """
    print('Starting validation...')

    # Set device to GPU if available, otherwise fallback to CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create a DataLoader for the validation dataset
    test_loader = torch.utils.data.DataLoader(
        listDataset(
            val_list, 
            config["dataset_name"],
            shuffle=False,  
            transform=transforms.Compose([
                transforms.ToTensor(),  # Convert PIL images to tensors
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                                     std=[0.229, 0.224, 0.225]),
            ]),
            train=False  # Set to validation mode
        ),
        batch_size=config["batch_size"]  # Batch size from the configuration
    )

    # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    model.eval()

    metrics = Metrics(device)  # Initialize custom metrics container

    # Disable gradient computation for faster validation and reduced memory usage
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            # Move images and targets to the appropriate device
            img = img.to(device)
            target = target.to(device).type(torch.FloatTensor)
            
            # Run forward pass to generate predictions
            output = model(img).to(device)

            # Update other metrics (e.g., RMSE, MAPE, R²) for this batch
            metrics.update(output, target)

    # Compute final metric values across the dataset
    mae = metrics.MAELoss()
    rmse = metrics.RMSELoss()
    mape = metrics.MAPELoss()
    r2 = metrics.R2Loss()

    # Print metrics in a formatted manner for better readability
    print()
    print(f' * MAE:  {mae:.3f}')
    print(f' * RMSE: {rmse:.3f}')
    print(f' * MAPE: {mape:.3f}')
    print(f' * R²:   {r2:.3f}')

    return mae
