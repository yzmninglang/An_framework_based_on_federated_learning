from itertools import chain
import torch
import torch.nn as nn
import numpy as np
from args import args_parser


class Net(nn.Module):
    def __init__(self, args):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        # self.layer1.weight.data.normal_(0, 0.1)  # 权重初始化 (均值为0，方差为0.1的正态分布)
        super(Net, self).__init__()
        self.args = args
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 988, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.args.n_action)
        )

    def forward(self, x):  # 定义forward函数 (x为状态)
        x = x.reshape(-1, 1, self.args.n_state)
        x = self.conv(x)
        # print(x.shape)
        x = x.reshape(-1, 32 * 988)
        x = self.fc(x)

        return x


def weight_init(m):  # 根据网络层的不同定义不同的初始化方式
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self, args):  # 定义DQN的一系列属性
        self.args = args
        self.eval_net, self.target_net = Net(self.args).to(self.args.device), Net(self.args).to(self.args.device)  # 利用Net创建两个神经网络: 评估网络EN和目标网络TN
        # self.eval_net.apply(weight_init)

        self.learn_step_counter = 0  # 用于 target 更新计时
        self.memo_idx = 0  # 记忆库记数
        self.memory = np.zeros((self.args.MEMORY_CAPACITY, self.args.n_state * 2 + 2))  # 初始化记忆库，是一个矩阵，即为5000*3952
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.args.lr, weight_decay=1e-4)  # torch 的优化器Adam
        self.loss_func = nn.MSELoss().to(self.args.device)  # 误差公式,使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):  # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(self.args.device)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        # 这里只输入一个 sample，第一维为batch_size = 1
        if np.random.random() < self.args.epsilon:  # 生成一个在[0, 1)内的随机数，如果小于self.args.epsilon，选择最优动作
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value.cpu(), 1)[1].data.numpy()[0]  # return the argmax
        else:  # 选随机动作()随机选择换哪一个位置
            action = np.random.randint(0, self.args.n_action)

        return action

    def store_transition(self, s, a, r, s_):  # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.array(list(chain(s, [a, r], s_)))  # 在水平方向上拼接数组
        # 如果记忆库满了, 就覆盖老数据
        self.memory[self.memo_idx, :] = transition  # 置入transition
        self.memo_idx = (self.memo_idx + 1) % self.args.MEMORY_CAPACITY  # 获取transition要置入的行数

    def learn(self):  # 定义学习函数(记忆库达到一定数量后便开始学习)
        # target net 参数更新
        if self.learn_step_counter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.args.MEMORY_CAPACITY, self.args.B)  # 随机抽取，可能会重复
        b_memory = self.memory[sample_index, :]  # 抽取self.args.B个索引对应的self.args.B个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :self.args.n_state]).to(self.args.device)
        b_a = torch.LongTensor(b_memory[:, self.args.n_state:self.args.n_state + 1].astype(int)).to(self.args.device)
        b_r = torch.FloatTensor(b_memory[:, self.args.n_state + 1:self.args.n_state + 2]).to(self.args.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.args.n_state:]).to(self.args.device)

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.args.gamma * q_next.max(1)[0].view(self.args.B, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()  # 更新评估网络的所有参数


if __name__ == '__main__':
    args = args_parser()
    dqn = DQN(args)
    input = torch.ones((64, 3953)).to(args.device) # 将数据转化为合适的形式：cuda，cpu
    output = dqn.eval_net(input)
    print(output.shape)
    for k in dqn.eval_net.state_dict().keys():
        print(k)
