import torch
class ot_loss():
    def __init__(self, sigma, c_size, stride):
        super(ot_loss, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        # coordinate is same to image space, set to constant since crop size is same
        # 实际上就成了（512,8），（4...508)
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32) + stride / 2
                                 # 在0-轴上增加维数
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, points):
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)
        if len(all_points) > 0:
            # 取第i列，转化为列向量
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            # *:hadamard product
            # self.cood是这个点的坐标（i.e.,arrange出来的），x,y是annotation的坐标
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis # 不同列相加了
            dis = dis.view((dis.size(0), -1))
    # 按每张图的点分
            dis_list = torch.split(dis, num_points_per_image)
            for dis in dis_list:
                if len(dis) > 0:
                    dist = -dis / (2.0 * self.sigma ** 2)
                    ot = torch.softmax(dist,dim = 0)*torch.exp(torch.sqrt(dis))

                    # 后验for all points
                else:
                    ot = None
                print(ot)
A = ot_loss(2,16,8)
import numpy as np
B = torch.from_numpy(np.array([[2,2],[1,1]],dtype = np.float32))
A.forward([B,B])
# A = torch.ones((2,4))
# B = torch.ones((3,4))
# print(A+B)