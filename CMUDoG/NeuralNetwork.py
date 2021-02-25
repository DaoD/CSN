import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import Dataset
from Metrics import Metrics
from tqdm import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self):
        raise NotImplementedError

    def train_step(self, data):
        with torch.no_grad():
            batch_u, batch_r, batch_p, batch_y = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()
        logits = self.forward(batch_u, batch_r, batch_p)
        loss = self.loss_func(logits, target=batch_y)
        loss.backward()
        self.optimizer.step()
        # print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(), batch_y.size(0)))  # , accuracy, corrects
        return loss, batch_y.size(0)

    def fit(self, X_train_utterances, X_train_responses, X_train_personas, y_train, X_dev_utterances, X_dev_responses, X_dev_personas, y_dev):
        if torch.cuda.is_available():
            self.cuda()

        dataset = Dataset(X_train_utterances, X_train_responses, X_train_personas, y_train)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        # self.loss_func = nn.CrossEntropyLoss()
        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0

            self.train()
            with tqdm(total=len(y_train), ncols=90) as pbar:
                for i, data in enumerate(dataloader):
                    loss, batch_size = self.train_step(data)
                    pbar.set_postfix(lr=self.args.learning_rate, loss=loss.item())

                    if i > 0 and i % 2000 == 0:
                        self.evaluate(X_dev_utterances, X_dev_responses, X_dev_personas, y_dev)
                        self.train()

                    if epoch >= 1 and self.patience >= 3:
                        # tqdm.write("Reload the best model...")
                        self.load_state_dict(torch.load(self.args.save_path))
                        self.adjust_learning_rate()
                        self.patience = 0

                    if self.init_clip_max_norm is not None:
                        utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                    pbar.update(batch_size)
                    avg_loss += loss.item()
            cnt = len(y_train) // self.args.batch_size + 1
            tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
            self.evaluate(X_dev_utterances, X_dev_responses, X_dev_personas, y_dev)
            tqdm.write("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        # tqdm.write("Decay learning rate to: " + str(self.args.learning_rate))

    def evaluate(self, X_dev_utterances, X_dev_responses, X_dev_personas, y_dev, is_test=False):
        y_pred = self.predict(X_dev_utterances, X_dev_responses, X_dev_personas)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, y_dev):
                output.write(str(score) + '\t' + str(label) + '\n')

        result = self.metrics.evaluate_all_metrics()

        if not is_test and result[0] + result[1] + result[2] > self.best_result[0] + self.best_result[1] + self.best_result[2]:
            # tqdm.write("save model!!!")
            self.best_result = result
            tqdm.write("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
            self.logger.info("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
            self.patience = 0
            torch.save(self.state_dict(), self.args.save_path)
        else:
            self.patience += 1

        if is_test:
            print("Evaluation Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (result[0], result[1], result[2]))

    def predict(self, X_dev_utterances, X_dev_responses, X_dev_personas):
        self.eval()
        y_pred = []
        dataset = Dataset(X_dev_utterances, X_dev_responses, X_dev_personas)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                batch_u, batch_r, batch_l = (item.cuda() for item in data)
                logits = self.forward(batch_u, batch_r, batch_l)
                y_pred.append(logits.data.cpu().numpy().reshape(-1))
        y_pred = np.concatenate(y_pred, axis=0).tolist()
        return y_pred

    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available():
            self.cuda()
