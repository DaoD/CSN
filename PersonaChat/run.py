import time
import argparse
import pickle
import random
import numpy as np
import torch
import logging
from CSN import CSN
import os

task_dic = {
    'both_original': "./data/both_original/",
    'both_revised': "./data/both_revised/",
}

# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='both_original',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--level",
                    default='word',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--batch_size",
                    default=15,
                    type=int,
                    help="The batch size.")
parser.add_argument("--gru_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--emb_size",
                    default=400,
                    type=int,
                    help="The embedding size")
parser.add_argument("--learning_rate",
                    default=1e-3,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--gamma",
                    default=0.3,
                    type=float,
                    help="Threshold.")
parser.add_argument("--decay",
                    default=0.9,
                    type=float,
                    help="Decay rate.")
parser.add_argument("--epochs",
                    default=5,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")
args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
args.save_path += args.task + '.' + CSN.__name__ + "." + args.level + "." + str(int(args.gamma * 10)) + "." + str(int(args.decay * 10)) + ".pt"
args.score_file_path = task_dic[args.task] + CSN.__name__ + "." + args.level + "." + str(int(args.gamma * 10)) + "." + str(int(args.decay * 10)) + "." + args.score_file_path
args.log_path += args.task + '.' + CSN.__name__ + "." + args.level + "." + str(int(args.gamma * 10)) + "." + str(int(args.decay * 10)) + ".log"

logging.basicConfig(filename=args.log_path, level=logging.INFO)
logger = logging.getLogger(__name__)

print(args)
print("Task: ", args.task)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def train_model():
    path = task_dic[args.task]
    X_train_utterances, X_train_responses, X_train_personas, _, _, _, y_train = pickle.load(file=open(path + "train.pkl", 'rb'))
    X_dev_utterances, X_dev_responses, X_dev_personas, _, _, _, y_dev = pickle.load(file=open(path + "valid.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open("./data/vocab_and_embeddings.pkl", 'rb'))
    word_embeddings = torch.FloatTensor(word_embeddings)
    model = CSN(word_embeddings, args=args, logger=logger)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    model.fit(
        X_train_utterances, X_train_responses, X_train_personas, y_train,
        X_dev_utterances, X_dev_responses, X_dev_personas, y_dev
    )

def test_model():
    path = task_dic[args.task]
    X_test_utterances, X_test_responses, X_test_personas, _, _, _, y_test = pickle.load(file=open(path + "test.pkl", 'rb'))
    vocab, word_embeddings = pickle.load(file=open("./data/vocab_and_embeddings.pkl", 'rb'))
    word_embeddings = torch.FloatTensor(word_embeddings)
    model = CSN(word_embeddings, args=args, logger=logger)
    model.load_model(args.save_path)
    model.evaluate(X_test_utterances, X_test_responses, X_test_personas, y_test, is_test=True)

if __name__ == '__main__':
    start = time.time()
    set_seed()
    if args.is_training:
        train_model()
        # test_model()
    else:
        print("test")
        test_model()
        # test_adversarial()
    end = time.time()
    print("use time: ", (end - start) / 60, " min")
