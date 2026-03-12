import numpy as np
import random
from collections import Counter


class Environment(object):
    def __init__(self, args):  # 指定UE的编号0~9，建立对应的环境
        self.args = args
        self.feature = [[] for _ in range(self.args.n_client)]  # UE当前的缓存状态：矩阵
        for i in range(self.args.n_client):  #10个client，每个client内部有3952，3952是指电影的总数（id号）,刚开始的时候系统没有任何缓存
            # +1使索引与内容编号直接对应：1~3952
            self.feature[i] = list(np.zeros(self.args.n_content + 1, dtype=np.int32))  # UE当前的缓存状态

        self.cache_content = [[] for _ in range(self.args.n_client)] # UE当前的缓存Sn
        for i in range(self.args.n_client):
            # 随机选择self.args.Sn个文件缓存，记录它们的索引
            self.cache_content[i] = random.sample(range(1, self.args.n_content + 1), self.args.Sn)

        self.reward_his = []  # 第一步到当前步每一步获得的reward
        self.request_his_local = [[] for _ in range(self.args.n_client)]  # 记录每个时隙每个基站的请求序列

    def reset(self):
        for q in range(self.args.n_client):
            for c in range(1, self.args.n_content + 1):
                # 如果
                if c in self.cache_content[q]:
                    self.feature[q][c] = 1

        initial_state = self.feature

        return initial_state

    def step(self, action, request, idx):  # Chit_local
        """输入参数action：表示选择的动作，值为0~self.args.n_content，
            0表示不替换，其他数值表示替换对应的内容"""
        #idx是指第几个基站
        #request表示此次动作时有接下来50个影评


        # 所有历史请求中请求次数最高的内容
        # 如果当前还没有请求历史，则随机选择一个
        if not len(self.request_his_local[idx]):
            request_counts = Counter(self.request_his_local[idx])  # 统计从第一步到上一步所有请求中每个内容被请求的次数
            choose_content = random.sample(range(1, self.args.n_content + 1), 1)[0]
        #如果已经有了请求历史，则通过对请求历史的观察求出其中的最大值
        else:
            request_counts = Counter(self.request_his_local[idx])  # 统计从第一步到上一步所有请求中每个内容被请求的次数
            choose_content = request_counts.most_common(1)[0][0]  # 找出所有历史请求中请求次数最多的1个内容的视频Id

        if action != 0:  # 要替换一个内容,并且给出了替换的位置（100个中选一个）

            # 将第idx基站的3952数据量对应的第在当前Sn中想要替换的第action-1的数据对应的index进行置0
            self.feature[idx][self.cache_content[idx][action - 1]] = 0

            # 如果选择放进去的内容在缓存中，则重新选择当前步请求次数次高的内容并以此类推
            rank = 1
            while True:
                if choose_content in self.cache_content[idx]:  #如果该次选择的标签是Sn出现的一个标签
                    counts_list = Counter(request_counts).most_common()  # 将对请求次数的统计转化为列表
                     #历史标签不可能只有一个，一直往下选，流行度第一的选不了就选第二的,如果满足下面条件，说明历史request的所有数据均被缓存，这次就不替换
                    if len(counts_list) <= rank: 
                        break
                    choose_content = counts_list[rank][0]
                    #choos_content在上一行代码发生改变，在下一次判断时，一旦不满足choose_content在其中就跳出if，进入else
                    rank += 1
                else:
                    break

            self.cache_content[idx][action - 1] = choose_content  # 替换Sn中action-1的位置所对应的内容

        self.feature[idx][choose_content] = 1
        s_ = self.feature[idx]

        # reward
        hit_times = 0
        for c in request:
            # 如果此次的50个电影中，刚好其中有一部分的刚好在Sn中，则认为其hit_time增加
            if c in self.cache_content[idx]:
                hit_times += 1
        #将这次request得到的数据加载request的历史记录后面
        self.request_his_local[idx].extend(request)  

        if len(request) == 0:
            reward = 0
        else:
            reward = hit_times / len(request)  # 当前步的reward为本次的缓存命中率
        self.reward_his.append(reward)  # 记录每一步的reward

        return s_, reward, hit_times
