import argparse

import torch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_request', type=int, default=6000,
                        help="number of content requests received in a time slot")
    parser.add_argument('--n_client', type=int, default=10,
                        help="number of clients")
    parser.add_argument('--n_content', type=int, default=3952,
                        help='number of contents')
    parser.add_argument('--Sn', type=int, default=100,
                        help='storage size')
    parser.add_argument('--T', type=int, default=100,
                        help="number of time slots")
    parser.add_argument('--interval', type=int, default=50,
                        help='size of episode: Dt in the paper PCC')

    parser.add_argument('--B', type=int, default=64,
                        help="batch size")
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--epsilon', type=float, default=0.9,
                        help='epsilon-greedy policy parameter')  # 最优选择动作百分比
    parser.add_argument('--gamma', type=float, default=0.85,
                        help='discount factor')  # 奖励递减参数
    # parser.add_argument('--TARGET_REPLACE_ITER', type=int, default=50,
    #                     help='update frequency of target network')
    parser.add_argument('--MEMORY_CAPACITY', type=int, default=5000,
                        help='capacity of replay buffer')
    parser.add_argument('--n_action', type=int, default=100 + 1,  # Sn + 1
                        help='number of actions')
    parser.add_argument('--n_state', type=int, default=3952 + 1,  # n_content + 1
                        help='number of states')

    parser.add_argument('--n_base', type=int, default=4,
                        help='number of base layers')
    parser.add_argument('--Pa', type=float, default=0.5,
                        help='switching factor of PCC')

    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = args_parser()
    print(args)
