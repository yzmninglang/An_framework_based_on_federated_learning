from data_processing import send_weight_service
from data_processing import recive_weight_service
from data_processing import average_weights_service
from args import args_parser
# import pandas as pd


if __name__ == '__main__':
    args = args_parser()
    args.device = "cpu"
    cnt=0
    while True:
        # w_list=[]
        # request_len_list=[]
        # try:
        print("S:开始接收模型")
        # print("S:{}st ip".format(i))
        # 先接收下面发过来的模型
        w,request_len=recive_weight_service()
        print("S:开始聚合模型")
        w_avg=average_weights_service(w,request_len)
        print("S:开始分发模型")
        send_weight_service(w_avg)
    # except:
        #     continue
