import pandas as pd
import numpy as np
import torch
import copy
import pickle
import socket
# import pickle
from args import args_parser

np.random.seed(42)


# 将rating中的timestamp依次减去(最小值-1)，timestamp变为最小值为1
def normal(timestamp):
    # timestamp中最小值为956703952
    data = timestamp - 956703931
    return data


def get_dataset():
    # load ratings file in the format of <userID, movieID, rating, timestamp>
    ratings = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', header=None, names=['user_id', 'movie_id', 'rating',
                                                                                    'timestamp'], engine='python')

    # rating: normalise to 0, 1/4, 2/4, 3/4 and 1, respectively
    rating_normalised = {1: 0, 2: 1 / 4, 3: 2 / 4, 4: 3 / 4, 5: 1}
    ratings['rating'] = ratings['rating'].map(rating_normalised)
    ratings = ratings.sort_values(by='timestamp')
    ratings['timestamp'] = ratings['timestamp'].apply(normal)

    # load users file in the format of <userID, gender, age, occupation, zip code>
    # age: Normalize to 0, 1/6, 2/6, 3/6, 4/6, 5/6, 1, respectively
    # gender：change 'F', 'M' into '1', '0'
    users = pd.read_csv('./data/ml-1m/users.dat', sep='::', header=None,
                        names=['user_id', 'gender', 'age', 'occupation',
                               'zipcode'], engine='python')
    gender_to_int = {'F': 1, 'M': 0}
    users['gender'] = users['gender'].map(gender_to_int)
    age_normalised = {1: 0, 18: 1 / 6, 25: 2 / 6, 35: 3 / 6, 45: 4 / 6, 50: 5 / 6, 56: 1}
    users['age'] = users['age'].map(age_normalised)
    occupation_normalised = {0: 0, 1: 1 / 20, 2: 2 / 20, 3: 3 / 20, 4: 4 / 20, 5: 5 / 20, 6: 6 / 20, 7: 7 / 20,
                             8: 8 / 20, 9: 9 / 20,
                             10: 10 / 20, 11: 11 / 20, 12: 12 / 20, 13: 13 / 20, 14: 14 / 20, 15: 15 / 20, 16: 16 / 20,
                             17: 17 / 20,
                             18: 18 / 20, 19: 19 / 20, 20: 1}
    users['occupation'] = users['occupation'].map(occupation_normalised)

    # load movies file in the format of <movieID, title, genres>
    movies = pd.read_csv('./data/ml-1m/movies.dat', sep='::', header=None, names=['movie_id', 'title', 'genres'],
                         engine='python')

    return ratings, users, movies


def get_feature_label(client_request, users):
    """obtain the feature vector: ratings|age|gender
       and the corresponding category label y: y=1, when rating>=3/4 (normalised); y=0, else.
    """
    num = client_request.shape[0]  # no. of requests for each client
    user_arr = users.to_numpy()  # convert the 'users' dataframe into np.array
    feature = np.zeros((num, 8))  # user_id|movie_id|rating|gender|age|label|timestamp

    for i in range(num):
        feature[i, 0] = client_request[i, 0]  # user id
        feature[i, 1] = client_request[i, 1]  # movie id
        feature[i, 2] = client_request[i, 2]  # ratings
        feature[i, 3] = user_arr[client_request[i, 0] - 1, 1]  # gender
        feature[i, 4] = user_arr[client_request[i, 0] - 1, 2]  # age, client_request[i,0] is the user id,
        # client_request[i,0]-1 is corresponding to the index of that user_id
        feature[i, 5] = user_arr[client_request[i, 0] - 1, 3]  # occupation
        if client_request[i, 2] >= 3 / 4:  # y=1, when rating >=3/4 (normalised); y=0, else.
            feature[i, 6] = 1  # put the labels into the last column of feature matrix
        feature[i, 7] = client_request[i, 3]

    return feature


def sampling():
    """divide the dataset into 10 groups, each group with 604 users' data
        8 groups of data will be used for training, 2 groups for testing
        interpret each request as if it coming from a separate user, i.e, each group data from 604 users' request
    """
    ratings, users, movies = get_dataset()

    N = 6040  # total no. users
    n = 10  # no. clients
    sample = pd.DataFrame(
        columns=['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation', 'label', 'timestamp'])

    all_idxs = [i for i in range(len(ratings))]
    user_group = {}
    client_request = np.zeros(n, dtype=object)
    feature = np.zeros(n, dtype=object)

    for i in range(n):
        df = pd.DataFrame(columns=['user_id', 'movie_id', 'rating', 'timestamp'])
        for j in range(N - i, 9 - i, -10):  # including 604 users
            df_j = ratings.groupby(['user_id']).get_group(j)  # get user j's data
            df = pd.concat([df, df_j], sort=False)  # concatenate 604 users' data, namely SBS i's all data

        client_request[i] = df.to_numpy()  # convert each client's data (including 604 users) into array
        user_group[i] = set(all_idxs[0:len(client_request[i])])
        all_idxs = list(set(all_idxs) - user_group[i])

        feature[i] = get_feature_label(client_request[i], users)
        sample_i = pd.DataFrame(feature[i], columns=['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation',
                                                     'label', 'timestamp'])
        sample = pd.concat([sample, sample_i])

    sample = sample.astype({'user_id': 'int64', 'movie_id': 'int64', 'rating': 'float64',
                            'gender': 'int64', 'age': 'float64', 'occupation': 'float64', 'label': 'int64',
                            'timestamp': 'int64'})
    sample.to_csv('./sample.csv', sep='\t', header=True, encoding='latin-1', index=False,
                  columns=['user_id', 'movie_id', 'rating', 'gender', 'age', 'occupation', 'label', 'timestamp'])

    print('saved to SAMPLE_CSV')

    f1_name = 'user_group.pkl'
    save_dict(user_group, f1_name)

    return sample, user_group


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    n_classes = train_labels.max()
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    print(label_distribution)
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(1, n_classes + 1)]
    # (K, ...) 记录K个类别对应的样本索引集合

    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应的样本索引集合
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def recive_weight_client():
    # w_avg = copy.deepcopy(w[0])
    # data=[w_avg,weight]
    # 建立套接字
    print("C:Reciving Global Model.....\r\n")
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = ('', 7789)
    tcp_server_socket.bind(address)
    tcp_server_socket.listen(12)
    client_socket, clientAddr = tcp_server_socket.accept()
    print("{} connect ....\r\n".format(clientAddr))
    data = []
    while True:
        packet = client_socket.recv(4096)
        if not packet: break
        data.append(packet)
    recv=pickle.loads(b"".join(data))
    # print(recv)
    client_socket.close()
    w=recv
    return w

def send_weight_client(w, weight):
    # 收集分散的局部模型
    print("C:Send Local Model.....\r\n")
    data=[w,weight]
    dict_dump = pickle.dumps(data)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_addr = ("192.168.1.13", 7789)
    tcp_socket.connect(server_addr)
    tcp_socket.send(dict_dump)
    tcp_socket.close()

def send_weight_service(w):
    # 下方全局模型
    print("S:Send Global Model.....\r\n")
    dict_dump = pickle.dumps(w)
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip_addr=["192.168.1.15","192.168.1.16"]
    for _ in ip_addr:
        server_addr = (_, 7789)
        tcp_socket.connect(server_addr)
        tcp_socket.send(dict_dump)
        tcp_socket.close()
def recive_weight_service():
    # w_avg = copy.deepcopy(w[0])
    # data=[w_avg,weight]
    # 建立套接字
    print("S:Recive local Model.....\r\n")
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    address = ('', 7789)
    tcp_server_socket.bind(address)
    tcp_server_socket.listen(128)
    w_avg=[]
    request_len=[]
    cnt=0
    while True:
        if cnt==2:
            break
        client_socket, clientAddr = tcp_server_socket.accept()
        print("{} connect ..\r\n".format(clientAddr))
        data = []
        while True:
            packet = client_socket.recv(4096)
            if not packet: break
            data.append(packet)
        recv=pickle.loads(b"".join(data))
        # print(recv)
        client_socket.close()
        w_avg.append(recv[0])
        request_len.append(recv[1])
        cnt=cnt+1
    return w_avg, request_len
 
def average_weights(w, weight, args):
    """
    Returns the average of the weights.
    w是本地模型
    """
    w_avg = copy.deepcopy(w[0])
    # 第key个参数
    for key in w_avg.keys():
        # 第i个基站，一共有十个
        for i in range(0, len(w)):
            if i == 0:
                # 如果是第一个基站，那么，w_avg不需要自加，因为上面已经有了赋值
                w_avg[key] = w_avg[key] * len(weight[i])
            else:
                # 如果不是第一个基站，那么w_avg的参数，是等于0...i的加权和
                w_avg[key] += w[i][key] * len(weight[i])
        # 最后求加权平均
        w_avg[key] = torch.div(w_avg[key], args.n_request)
    return w_avg

def average_weights_service(w, weight):
    """
    Returns the average of the weights.
    w是本地模型
    """
    w_avg = copy.deepcopy(w[0])
    # 第key个参数
    for key in w_avg.keys():
        # 第i个基站，一共有十个
        for i in range(0, len(w)):
            if i == 0:
                # 如果是第一个基站，那么，w_avg不需要自加，因为上面已经有了赋值
                w_avg[key] = w_avg[key] * weight[i]
            else:
                # 如果不是第一个基站，那么w_avg的参数，是等于0...i的加权和
                w_avg[key] += w[i][key] * weight[i]
        # 最后求加权平均
        w_avg[key] = torch.div(w_avg[key], sum(weight))
    return w_avg

# def average_weights_of_local_model(w):
#     """
#     :param w: historical local model parameters for client i
#     :return: the average of w
#     """
#     w_avg = copy.deepcopy(w[0])
#     for key in w_avg.keys():
#         for i in range(0, len(w)):
#             if i == 0:
#                 w_avg[key] = w_avg[key]
#             else:
#                 w_avg[key] += w[i][key]
#         w_avg[key] = torch.div(w_avg[key], len(w))
#
#     return w_avg


# 保存和读取字典文件
def save_dict(dict, f_name):
    f_save = open(f_name, 'wb')
    pickle.dump(dict, f_save)
    f_save.close()
    print('save to file.pkl')


def load_dict(f_name):
    f_read = open(f_name, 'rb')
    dict2 = pickle.load(f_read)
    # print(dict2)
    f_read.close()
    return dict2


if __name__ == '__main__':
    sampling()
