from caching_env import Environment
from model import DQN
from data_processing import average_weights
from args import args_parser
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# PCC


def main(args):
    start = time.time()  # 程序开始的时间
    chr_list = []  # 记录每个时隙中系统的缓存命中率

    # BUILD GLOBAL MODEL
    global_model = DQN(args)
    global_model.eval_net.train()

    # dqn = [DQN(args) for _ in range(args.n_client)]
    # dqn1 = [DQN(args) for _ in range(args.n_client)]

    # Build environment for every client
    ue_env = Environment(args)
    ue_env.reset()
    env = ue_env

    hit_times_T = 0
    local_weights = []  # 保存每个client在该时隙训练完后的本地模型
    local_weights1 = []
    for t in range(args.T):
        start1 = time.time()
        hit_times_t = []  # 保存每个client在该时隙中的命中次数
        env.request_his = []
        env.request_his_local = [[] for _ in range(args.n_client)]

        # 读取数据，该时隙所有基站的请求数据
        samples = pd.read_csv('order_data.csv', skiprows=args.n_request * t, nrows=args.n_request)
        request = [[] for _ in range(args.n_client)]  # 将产生的请求分到它所对应的小基站
        # request = [[], [], [], [], [], [], [], [], [], []]
        global_weights = copy.deepcopy(global_model.eval_net.state_dict())  # 得到全局模型参数
        for i in range(len(samples)):
            request_i = samples.iloc[i, 1]
            user_id = samples.iloc[i, 0]
            if user_id % args.n_client == 0:
                request[0].append(request_i)
            else:
                group_id = args.n_client - (user_id % args.n_client)
                request[group_id].append(request_i)

        # 对每一个小基站进行本地训练
        for idx in range(args.n_client):
            # local_model = dqn[idx]
            local_model = DQN(args)
            local_model.eval_net.train()
            local_model.eval_net.load_state_dict(global_weights)

            # local_model1 = dqn1[idx]
            local_model1 = DQN(args)
            local_model1.eval_net.train()
            if t >= 1:
                local_model1.eval_net.load_state_dict(local_weights1[idx])

            state = env.feature[idx]
            hit_times_list = []  # 保存当前client在本地训练中每个episode的命中次数

            iter1 = 0
            while args.interval * iter1 < len(request[idx]):
                P = np.random.random()
                # 将每一组数据分为多组进行len(request[idx]) // args.interval个回合的本地训练，每一小组的数据量为args.interval
                watchlist_i = request[idx][iter1 * args.interval:(iter1 + 1) * args.interval]
                if P >= args.Pa:  # Mg
                    action = local_model.choose_action(state)
                    state_, reward, hit_times = env.step(action, watchlist_i, idx)
                    hit_times_list.append(hit_times)
                    local_model.store_transition(state, action, reward, state_)
                    # 收集到一定数量requests后再进行学习
                    # if iter1 > 1:
                    local_model.learn()
                else:  # Ml
                    action = local_model1.choose_action(state)
                    state_, reward, hit_times = env.step(action, watchlist_i, idx)
                    hit_times_list.append(hit_times)
                    local_model1.store_transition(state, action, reward, state_)
                    # 收集到一定数量requests后再进行学习
                    # if iter1 > 1:
                    local_model1.learn()

                state = state_
                iter1 = iter1 + 1

            w = copy.deepcopy(local_model.eval_net.state_dict())
            w1 = copy.deepcopy(local_model1.eval_net.state_dict())
            if t == 0:
                local_weights.append(w)
                local_weights1.append(w1)
            else:
                local_weights[idx] = w
                local_weights1[idx] = w1

            hit_times_ue = sum(hit_times_list)  # client在该时隙总的命中次数
            hit_times_t.append(hit_times_ue)  # 记录每个client在该时隙的总命中次数

        # cache_hit_ratio1 = sum(hit_times_t) / args.n_request  # 该时隙系统的命中率
        hit_times_T = hit_times_T + sum(hit_times_t)
        cache_hit_ratio = hit_times_T / ((t + 1) * args.n_request)  # 优化目标：长期CHR
        if (t + 1) % 10 == 0:
            print('\n |The cache hit ratio of time slot: {} is : {:.2%}|'.format(t, cache_hit_ratio))
        print('Time slot', t, ' elapsed:', time.time() - start1)
        chr_list.append(cache_hit_ratio)

        # federated aggregation
        global_weights = average_weights(local_weights, request, args)
        global_model.eval_net.load_state_dict(global_weights)

    print('Time elapsed:', time.time() - start)

    return chr_list


if __name__ == '__main__':
    args = args_parser()
    args.device = "cuda:1"
    p=0.4
    m = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    chr = []
    for j in range(10):
        c_t = []
        for i in range(len(m)):
            args.n_request = m[i]
            cache_hit = main(args)
            # cache_hit = [i]
            c = [round(i * 100, 2) for i in cache_hit]  # 转换成%的数值并保留两位小数输出
            print(f'M={m[i]} | c={c}')
            c_t.append(c)
        print(f'第{j + 1}/10次运行 | c_t:{c_t}')
        chr.append(c_t)
    print(chr)
