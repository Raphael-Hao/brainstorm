# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import time
import torch.multiprocessing as mp
import torch


class Receiver(mp.Process):
    def __init__(self, conn, barrier: mp.Barrier, data_num, repeat):
        super(Receiver, self).__init__()
        self.conn = conn
        self.barrier = barrier
        self.data_num = data_num
        self.repeat = repeat

    def run(self):
        for _ in range(self.repeat):
            self.barrier.wait()
            for i in range(self.data_num):
                data = self.conn.recv()
            end_stamp = time.time()
            self.conn.send(end_stamp)

        self.barrier.wait()
        self.conn.close()


sender_conn, receiver_conn = mp.Pipe()
data_num = 10000
repeat = 20
barrier = mp.Barrier(2)
receiver = Receiver(receiver_conn, barrier, data_num, repeat)
receiver.start()
datas = [torch.randn((150000,), device="cuda") for i in range(data_num)]

for r_i in range(repeat):
    barrier.wait()
    start_stamp = time.time()
    for i, data in enumerate(datas):
        sender_conn.send(data)
    end_stamp = sender_conn.recv()
    print(
        f"{r_i}-th trial, time cost: {(end_stamp - start_stamp) * 1000 / data_num:.3f}"
    )

barrier.wait()
receiver.join()
sender_conn.close()

datas[0].storage()