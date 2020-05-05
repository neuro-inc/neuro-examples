import io

import torch
from main import Net
from PIL import Image
from torchvision import transforms


class SeldonModel:
    def __init__(self):
        self._model = Net()
        self._model.load_state_dict(
            torch.load("/storage/model.pkl", map_location=torch.device("cpu"))
        )
        self._model.eval()

    def predict(self, X, features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        data = transforms.ToTensor()(Image.open(io.BytesIO(X)))
        return self._model(data[None, ...]).detach().numpy()

    def send_feedback(self, features, feature_names, reward, truth):
        """
        Handle feedback

        Parameters
        ----------
        features : array - the features sent in the original predict request
        feature_names : array of feature names. May be None if not available.
        reward : float - the reward
        truth : array with correct value (optional)
        """
        print("Send feedback called")
        return []


seldon_model = SeldonModel
