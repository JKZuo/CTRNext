import time
import random
from torch import optim
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import joblib
from models import *


def calculate_acc(prob, label):
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)

def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)

def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0

def calculate_ndcg(prob, label, k):
    # prob (N, L), label (N), k: Top-K
    ndcg_scores = []
    
    for j in range(len(prob)):
        # Sort predictions in descending order and get top-K
        sorted_indices = np.argsort(prob[j])[::-1][:k]
        
        # Compute DCG (Discounted Cumulative Gain)
        dcg = 0
        for i, idx in enumerate(sorted_indices):
            if label[j] == idx:
                dcg += 1 / np.log2(i + 2)  # Note: i+2 because the index starts from 0
            
        # Compute IDCG (Ideal DCG) assuming perfect ranking
        ideal_sorted_indices = np.argsort(label[j])[::-1][:k]
        idcg = 0
        for i, idx in enumerate(ideal_sorted_indices):
            idcg += 1 / np.log2(i + 2)
        
        # Compute NDCG as DCG divided by IDCG
        if idcg == 0:
            ndcg = 0  # Avoid division by zero
        else:
            ndcg = dcg / idcg
        
        ndcg_scores.append(ndcg)
    
    # Calculate the average NDCG
    avg_ndcg = np.mean(ndcg_scores)
    
    return avg_ndcg


def CTR_samplingprob(probability_matrix, labels, num_negative_samples):
    num_labels, label_max = probability_matrix.shape[0], probability_matrix.shape[1] - 1  
    labels = labels.view(-1)  
    initial_labels = np.linspace(0, num_labels - 1, num_labels)  
    initial_probability = torch.zeros(size=(num_labels, num_negative_samples + len(labels)))  

    random_negative_indices = random.sample(range(1, label_max + 1), num_negative_samples)  
    while len([lab for lab in labels if lab in random_negative_indices]) != 0:
        random_negative_indices = random.sample(range(1, label_max + 1), num_negative_samples)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    for k in range(num_labels):
        for i in range(num_negative_samples + len(labels)):
            if i < len(labels):
                initial_probability[k, i] = probability_matrix[k, labels[i]]
            else:
                initial_probability[k, i] = probability_matrix[k, random_negative_indices[i - len(labels)]]

    return torch.FloatTensor(initial_probability), torch.LongTensor(initial_labels)


class CTRDataset(data.Dataset):
    def __init__(self, trajectories, matrices1, vectors, labels, lengths):
        self.trajectories = trajectories
        self.matrices1 = matrices1
        self.vectors = vectors
        self.labels = labels
        self.lengths = lengths

    def __getitem__(self, index):
        trajectory = self.trajectories[index].to(device)
        matrices1 = self.matrices1[index].to(device)
        vector = self.vectors[index].to(device)
        label = self.labels[index].to(device)
        length = self.lengths[index].to(device)
        return trajectory, matrices1, vector, label, length

    def __len__(self):
        return len(self.trajectories)


class Trainer:
    def __init__(self, model, record):
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = 1
        self.learning_rate = 3e-3
        self.num_epoch = 100
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0

        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens
        
        self.dataset = CTRDataset(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            val_NDCG_5, test_NDCG_5, val_NDCG_10, test_NDCG_10, val_NDCG_20, test_NDCG_20 = 0, 0, 0, 0, 0, 0
            MRR_valid, MRR_test = 0, 0
            mAP20_val, mAP20_test  = 0, 0

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                max_len = 100 
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1): 
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len) 

                    if mask_len <= person_traj_len[0] - 2: 
                        prob_sample, label_sample = CTR_samplingprob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  
                        valid_size += person_input.shape[0]
                        acc_valid += calculate_acc(prob, train_label)
                        cpu_label = train_label.detach().cpu().numpy()
                        cpu_prob = prob.detach().cpu().numpy()
                        MRR_valid += MRR_metric_last_timestep(cpu_label, cpu_prob) 
                        mAP20_val += mAP_metric_last_timestep(cpu_label, cpu_prob, k=20)
                        val_NDCG_5 += calculate_ndcg(cpu_prob, cpu_label, k=5)
                        val_NDCG_10 += calculate_ndcg(cpu_prob, cpu_label, k=10)
                        val_NDCG_20 += calculate_ndcg(cpu_prob, cpu_label, k=20)

                    elif mask_len == person_traj_len[0]:
                        test_size += person_input.shape[0]
                        acc_test += calculate_acc(prob, train_label)
                        cpu_label = train_label.detach().cpu().numpy()
                        cpu_prob = prob.detach().cpu().numpy()
                        MRR_test += MRR_metric_last_timestep(cpu_label, cpu_prob)
                        mAP20_test += mAP_metric_last_timestep(cpu_label, cpu_prob, k=20)
                        test_NDCG_5 += calculate_ndcg(cpu_prob, cpu_label, k=5)
                        test_NDCG_10 += calculate_ndcg(cpu_prob, cpu_label, k=10)
                        test_NDCG_20 += calculate_ndcg(cpu_prob, cpu_label, k=20)

                bar.update(1)  
            bar.close()             

            acc_valid = np.array(acc_valid) / valid_size
            MRR_valid = np.array(MRR_valid) / valid_size
            mAP20_val = np.array(mAP20_val) / valid_size
            val_NDCG5= np.array(val_NDCG_5) / valid_size
            val_NDCG10= np.array(val_NDCG_10) / valid_size
            val_NDCG20= np.array(val_NDCG_20) / valid_size
            print('epoch:{}, time:{:.0f}, valid_acc:{}, MRR:{:.3f}, mAP20:{:.3f}, NDCG5:{:.3f}, NDCG10:{:.3f}, NDCG20:{:.3f}'.format(self.start_epoch+t, 
                              time.time()-start, acc_valid, MRR_valid, mAP20_val, val_NDCG5, val_NDCG10, val_NDCG20))

            acc_test = np.array(acc_test) / test_size
            MRR_test = np.array(MRR_test) / test_size
            mAP20_test = np.array(mAP20_test) / test_size
            test_NDCG5 = np.array(test_NDCG_5) / test_size
            test_NDCG10 = np.array(test_NDCG_10) / test_size
            test_NDCG20 = np.array(test_NDCG_20) / test_size
            print('epoch:{}, time:{:.0f}, test_acc:{}, MRR:{:.3f}, mAP20:{:.3f}, NDCG5:{:.3f}, NDCG10:{:.3f}, NDCG20:{:.3f}'.format(self.start_epoch+t, 
                              time.time()-start, acc_test, MRR_test, mAP20_test, test_NDCG5, test_NDCG10, test_NDCG20))

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + t)


    def inference(self):
        user_ids = []
        for t in range(self.num_epoch):
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_valid, cum_test = [0, 0, 0, 0], [0, 0, 0, 0]

            for step, item in enumerate(self.data_loader):
                max_len = 100 
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0] + 1): 
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)

                    if mask_len <= person_traj_len[0] - 2:
                        continue

                    elif mask_len == person_traj_len[0] - 1:
                        acc_valid = calculate_acc(prob, train_label)
                        cum_valid += calculate_acc(prob, train_label)

                    elif mask_len == person_traj_len[0]:
                        acc_test = calculate_acc(prob, train_label)
                        cum_test += calculate_acc(prob, train_label)
                print(step, acc_valid, acc_test)

                if acc_valid.sum() == 0 and acc_test.sum() == 0:
                    user_ids.append(step)


if __name__ == '__main__':
    file = open('./data/TKY.pkl', 'rb')   
    file_data = joblib.load(file)
    [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
    mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(device), torch.FloatTensor(mat2t), torch.LongTensor(lens)

    part = 100
    trajs, mat1, mat2t, labels, lens = trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]
    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()
    hours = 24*7
    CTR = Model(t_dimension = hours+1, l_dimension = l_max+1, u_dimension = u_max+1, embedding_dimension = 50, ex = ex)
    num_params = 0
    
    for param in CTR.parameters():
        num_params += param.numel()
    print('Parameters of CTR model', num_params)

    load = False
    records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
    start = time.time()

    trainer = Trainer(CTR, records)
    trainer.train()

# nohup python train.py > train.log 2>&1 &
