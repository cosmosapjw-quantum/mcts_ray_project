# inference/inference_server.py
import ray
import torch
import time
from threading import Thread
from queue import Queue
from model import SmallResNet

@ray.remote
class InferenceServer:
    def __init__(self, batch_wait=0.01):
        self.model = SmallResNet()
        self.model.eval()
        self.queue = Queue()
        self.batch_wait = batch_wait
        Thread(target=self._batch_worker, daemon=True).start()

    def infer(self, state):
        future = Queue()
        self.queue.put((state, future))
        result = future.get()
        return result

    def _batch_worker(self):
        while True:
            states, futures = [], []
            start_time = time.time()

            while len(states) == 0 or ((time.time() - start_time) < self.batch_wait and not self.queue.empty()):
                try:
                    state, future = self.queue.get(timeout=self.batch_wait)
                    states.append(state)
                    futures.append(future)
                except:
                    pass  # timeout occurred, proceed to inference

            if states:
                inputs = torch.stack([torch.tensor(s.board).float() for s in states])
                with torch.no_grad():
                    policy_batch, value_batch = self.model(inputs)

                policy_batch = policy_batch.numpy()
                value_batch = value_batch.numpy()

                for future, policy, value in zip(futures, policy_batch, value_batch):
                    future.put((policy, value))
