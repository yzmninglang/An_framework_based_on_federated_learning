from caching_env import Environment
from model import DQN
from args import args_parser
import matplotlib.pyplot as plt
import pandas as pd
import copy
import time
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# DQN w/o collaboration


def main(args):
    start = time.time()  # 程序开始的时间
    chr_list = []  # 记录每个时隙中系统的缓存命中率

    # Build environment for every client
    ue_env = Environment(args)
    ue_env.reset()
    env = ue_env

    hit_times_T = 0
    local_weights1 = []
    # 默认为100个slot，每个时隙做一下几部
    # 1.读取6000条数据，安装每个数据分配到同一个基站的原则分配数据
        # 2.对每一个小基站进行训练
    for t in range(args.T):
        start1 = time.time()
        hit_times_t = []  # 保存每个client在该时隙中的命中次数
        env.request_his = []
        env.request_his_local = [[] for _ in range(args.n_client)]

        # 1. 读取数据，该时隙所有基站的请求数据:每次读6000samples
        samples = pd.read_csv('order_data.csv', skiprows=args.n_request * t, nrows=args.n_request)
        request = [[] for _ in range(args.n_client)]  # 将产生的请求分到它所对应的小基站
        # request = [[], [], [], [], [], [], [], [], [], []]
        # 6000条数据的samples
        for i in range(len(samples)):
            request_i = samples.iloc[i, 1]  #request_i是电影的id号
            user_id = samples.iloc[i, 0]    #观看电影的用户
            if user_id % args.n_client == 0:    #同一个用户分在同一个基站里面
                request[0].append(request_i)
            else:
                group_id = args.n_client - (user_id % args.n_client)    #同一个用户放在同一个基站里面
                request[group_id].append(request_i)

        # 2. 对每一个小基站进行本地训练
        # for idx in range(args.n_client):
        for idx in range(args.n_client):
            # idx=0
            local_model1 = DQN(args)
            local_model1.eval_net.train()
            # 如果slot不为1,是针对外圈循环来说的，如果不为1，则先载入已经存在的权重，在训练
            if t >= 1:
                local_model1.eval_net.load_state_dict(local_weights1[idx])

            state = env.feature[idx] #UE当前的缓存状态
            hit_times_list = []  # 保存当前client在本地训练中每个episode的命中次数

            iter1 = 0
            # interval是50，也就是说 while循环需要进行12次，即iter1最后为12
            while args.interval * iter1 < len(request[idx]): #request是一个10*600的矩阵
                # 将每一组数据分为多组进行len(request[idx]) // args.interval个回合的本地训练，每一小组的数据量为args.interval
                # watchlist_i对应的是这次训练使用的50个电影缓存
                watchlist_i = request[idx][iter1 * args.interval:(iter1 + 1) * args.interval]
            
                action = local_model1.choose_action(state) #根据现在的state和概率选择是否更换Sn（size:100）中的某一个位置的内容
                state_, reward, hit_times = env.step(action, watchlist_i, idx)
                hit_times_list.append(hit_times)
                local_model1.store_transition(state, action, reward, state_)
                # 收集到一定数量requests后再进行学习
                # if iter1 > 1:
                local_model1.learn()

                state = state_
                iter1 = iter1 + 1
            # 复制了这个参数
            w1 = copy.deepcopy(local_model1.eval_net.state_dict())
            # 将每次训练得到的参数组成一个list
            if t == 0:
                local_weights1.append(w1)
            else:
                local_weights1[idx] = w1
            # hit_times_list保存了12个数据，刚好1个slot
            hit_times_ue = sum(hit_times_list)  # 计算client在该时隙总的命中次数
            hit_times_t.append(hit_times_ue)  # 记录每个client在该时隙的总命中次数

            # cache_hit_ratio1 = sum(hit_times_t) / args.n_request  # 该时隙系统的命中率
            hit_times_T = hit_times_T + sum(hit_times_t)
            cache_hit_ratio = hit_times_T / ((t + 1) * args.n_request)  # 优化目标：长期CHR
            if (t + 1) % 10 == 0:
                print('\n |The cache hit ratio of time slot: {} is : {:.2%}|'.format(t, cache_hit_ratio))
            print('Time slot', t, ' elapsed:', time.time() - start1)
            chr_list.append(cache_hit_ratio)

    print('Time elapsed:', time.time() - start)

    return chr_list


if __name__ == '__main__':
    args = args_parser()
    # args.device = "cuda:0"
    # sn = [20, 40, 60, 80, 100, 120, 140, 160, 180]
    # chr = []
    # for j in range(10):
    #     c_t = []
    #     for i in range(len(sn)):
    #         args.Sn = sn[i]
    #         args.n_action = sn[i] + 1
    #         cache_hit = main(args)
    #         # cache_hit = [i]
    #         c = [round(i * 100, 2) for i in cache_hit]  # 转换成%的数值并保留两位小数输出
    #         print(f'Sn={sn[i]} | c={c}')
    #         c_t.append(c)
    #     print(f'第{j + 1}/5次运行 | c_t:{c_t}')
    #     chr.append(c_t)
    # print(chr)

    # args = args_parser()
    # t = [10, 30, 50, 70, 90, 110, 130, 150]
    # chr = []
    # for j in range(10):
    #     c_t = []
    #     for i in range(len(t)):
    #         args.T = t[i]
    #         cache_hit = main(args)
    #         # cache_hit = [i]
    #         c = [round(i * 100, 2) for i in cache_hit]  # 转换成%的数值并保留两位小数输出
    #         print(f'T={t[i]} | c={c}')
    #         c_t.append(c)
    #     print(f'第{j + 1}/10次运行 | c_t:{c_t}')
    #     chr.append(c_t)
    # print(chr)

    # args = args_parser()
    args.device = "cuda:1"
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

    # args = args_parser()
    # # args.device = "cuda:1"
    # # D = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    # D = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    # # D = [42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]
    # chr = []
    # for j in range(10):
    #     c_t = []
    #     for i in range(len(D)):
    #         args.interval = D[i]
    #         cache_hit = main(args)
    #         # cache_hit = [i]
    #         c = [round(i * 100, 2) for i in cache_hit]  # 转换成%的数值并保留两位小数输出
    #         print(f'D={D[i]} | c={c}')
    #         c_t.append(c)
    #     print(f'第{j + 1}/10次运行 | c_t:{c_t}')
    #     chr.append(c_t)
    # print(chr)
