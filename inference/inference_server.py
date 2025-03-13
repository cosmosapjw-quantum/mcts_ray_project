# inference/inference_server.py
import ray
from model import SmallResNet
import torch

@ray.remote
class InferenceServer:
    def __init__(self):
        self.model = SmallResNet()

    def infer(self, states):
        inputs = torch.stack([torch.tensor(s.board).float() for s in states])
        with torch.no_grad():
            policy, value = self.model(inputs)
        return policy.numpy(), value.numpy()