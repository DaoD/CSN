import argparse
import random
import numpy as np
import torch
import logging
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from CSN import CSN
from Metrics import Metrics
from file_dataset import FileDataset
from tqdm import tqdm
import os

task_config_dict = {
    'personachat': {
        'max_uttr_num': 15,
        'max_uttr_len': 20,
        'max_response_num': 20,
        'max_response_len' : 20,
        'max_persona_num': 5,
        'max_persona_len': 15,
        'max_word_length' : 18
    },
    'cmudog': {
        'max_uttr_num': 15,
        'max_uttr_len': 40,
        'max_response_num': 20,
        'max_response_len' : 40,
        'max_persona_num': 20,
        'max_persona_len': 40,
        'max_word_length' : 18
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=16,
                    type=int,
                    help="The batch size.")
parser.add_argument("--level",
                    default='word',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=16,
                    type=int,
                    help="The batch size.")
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
parser.add_argument("--task",
                    default="personachat",
                    type=str,
                    help="Task")
parser.add_argument("--gru_hidden",
                    default=300,
                    type=int,
                    help="The hidden size of GRU in layer 1")
parser.add_argument("--emb_size",
                    default=550,
                    type=int,
                    help="The embedding size")
parser.add_argument("--mode",
                    default="student",
                    type=str,
                    help="Training mode.")
parser.add_argument("--file_suffix",
                    default="self_original",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=15,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./model/",
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
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
result_path = "./output/" + args.task + "/"
if args.task == "cmudog":
    args.save_path += args.task + "." + CSN.__name__ + "." + args.mode + ".pt"
    args.score_file_path = result_path + CSN.__name__ + "." + args.mode + "." + args.score_file_path
    args.log_path += args.task + "." + CSN.__name__ + "." + args.mode + ".log"
else:
    args.save_path += args.task + "." + args.file_suffix + "." + CSN.__name__ + "." + args.mode + ".pt"
    args.score_file_path = result_path + CSN.__name__ + "." + args.file_suffix + "." + args.mode + "." + args.score_file_path
    args.log_path += args.task + "." + args.file_suffix + "." + CSN.__name__  + "." + args.mode + ".log"

device = torch.device("cuda:0")

logging.basicConfig(filename=args.log_path, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info(args)

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_model():
    train_data = "./data/" + args.task + "_processed/processed_train_" + args.file_suffix + ".txt" 
    test_data = "./data/" + args.task + "_processed/processed_valid_" + args.file_suffix + ".txt" 
    word_embeddings = torch.FloatTensor(torch.load("./data/embed_vocab_" + args.task + "/word_embeddings.pt"))
    char_embeddings = torch.FloatTensor(torch.load("./data/embed_vocab_" + args.task + "/char_embeddings.pt"))
    model = CSN(word_embeddings, char_embeddings, args, mode=args.mode)
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info('* number of parameters: %d' % n_params)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    fit(model, train_data, test_data)

def train_step(model, train_data, ce):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    y_pred = model.forward(train_data)
    batch_y = train_data["labels"]
    loss = 0.0
    loss += ce(y_pred, batch_y)
    return loss

def fit(model, X_train, X_test):
    train_dataset = FileDataset(X_train, "./data/embed_vocab_" + args.task + "/vocab.txt", "./data/embed_vocab_" + args.task + "/char_vocab.txt", task_config_dict[args.task]['max_uttr_num'], task_config_dict[args.task]['max_uttr_len'], task_config_dict[args.task]['max_persona_num'], task_config_dict[args.task]['max_persona_len'], task_config_dict[args.task]['max_response_num'], task_config_dict[args.task]['max_response_len'], task_config_dict[args.task]['max_word_length'], args.task)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    ce = torch.nn.CrossEntropyLoss()

    one_epoch_step = len(train_dataset) // args.batch_size
    best_result = [0.0, 0.0, 0.0]
    patience = 0
    
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        avg_loss = 0
        model.train()
        epoch_iterator = tqdm(train_dataloader, ncols=130)
        for i, training_data in enumerate(epoch_iterator):
            loss_ce = train_step(model, training_data, ce)
            loss_ce = loss_ce.mean()
            loss = loss_ce
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, ce=loss_ce.item())

            if i > 0 and i % (one_epoch_step // 5) == 0:
                best_result, patience = evaluate(model, X_test, best_result, patience)
                model.train()

            if epoch >= 1 and patience >= 3:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                model_state_dict = torch.load(args.save_path)
                model.load_state_dict({"module." + k: v for k, v in model_state_dict.items()})
                patience = 0
            avg_loss += loss.item()

        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))

def evaluate(model, X_test, best_result, patience, is_test=False):
    y_pred, y_label = predict(model, X_test)
    metrics = Metrics(args.score_file_path)

    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')

    result = metrics.evaluate_all_metrics()

    if not is_test and result[0] + result[1] + result[2] > best_result[0] + best_result[1] + best_result[2]:
        # tqdm.write("save model!!!")
        best_result = result
        tqdm.write("Best Result: R1: %.4f R2: %.4f R5: %.4f" % (best_result[0], best_result[1], best_result[2]))
        logger.info("Best Result: R1: %.4f R2: %.4f R5: %.4f" % (best_result[0], best_result[1], best_result[2]))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    else:
        patience += 1

    if is_test:
        print("Best Result: R1: %.4f R2: %.4f R5: %.4f" % (best_result[0], best_result[1], best_result[2]))
    
    return best_result, patience

def predict(model, X_test):
    model.eval()
    test_dataset = FileDataset(X_test, "./data/embed_vocab_" + args.task + "/vocab.txt", "./data/embed_vocab_" + args.task + "/char_vocab.txt", task_config_dict[args.task]['max_uttr_num'], task_config_dict[args.task]['max_uttr_len'], task_config_dict[args.task]['max_persona_num'], task_config_dict[args.task]['max_persona_len'], task_config_dict[args.task]['max_response_num'], task_config_dict[args.task]['max_response_len'], task_config_dict[args.task]['max_word_length'], args.task)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=130, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_test = model.forward(test_data)
            # y_pred_test = torch.stack(y_pred_test_list, dim=0).mean(dim=0)
            y_pred.append(y_pred_test.data.cpu().numpy().reshape(-1))
            y_tmp_label = test_data["labels"].data.cpu().numpy().tolist()
            y_label_one_hot = np.zeros((len(y_tmp_label), 20), dtype=np.int32)
            for i in range(len(y_tmp_label)):
                y_label_one_hot[i][y_tmp_label[i]] = 1
            y_label_one_hot = y_label_one_hot.reshape(-1)
            y_label.append(y_label_one_hot)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()
    return y_pred, y_label

def test_model():
    test_data = "./data/" + args.task + "_processed/processed_test_" + args.file_suffix + ".txt" 
    word_embeddings = torch.FloatTensor(torch.load("./data/embed_vocab_" + args.task + "/word_embeddings.pt"))
    char_embeddings = torch.FloatTensor(torch.load("./data/embed_vocab_" + args.task + "/char_embeddings.pt"))
    model = CSN(word_embeddings, char_embeddings, args, mode=args.mode)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    evaluate(model, test_data, [0.0, 0.0, 0.0], 0, is_test=True)

if __name__ == '__main__':
    set_seed(0)
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
