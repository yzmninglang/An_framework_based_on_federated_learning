import socket
import pickle

dict_x={1,2,3,4}
dict_dump = pickle.dumps(dict_x)

# 1.创建socket
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 2. 链接服务器
server_addr = ("192.168.1.16", 7789)
tcp_socket.connect(server_addr)

# 3. 发送数据
# send_data = input("请输入要发送的数据：")

tcp_socket.send(dict_dump)

# 4. 关闭套接字
tcp_socket.close()