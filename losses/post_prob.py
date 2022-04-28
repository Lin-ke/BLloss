import torch
import logging
from torch.nn import Module

class Post_Prob(Module):
    # c_size = 512,stride = 8,512/8 = 64
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0
        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
                                 # 在0-轴上增加维数
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
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
            prob_list = []
            ot_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        # mindis = relu dis
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        bg_dis = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dist = -dis / (2.0 * self.sigma ** 2)
                    # 后验for all points
                    prob = self.softmax(dist)
                    ot = 0.5*torch.sum(prob*torch.exp(0.1*(torch.sqrt(dis+0.000001))),dim=0)

                else:
                    prob = None
                    ot = None
                prob_list.append(prob)
                ot_list.append(ot)
        else:
            prob_list = []
            ot_list = []
            for _ in range(len(points)):
                prob_list.append(None)
                ot_list.append(None)
        return prob_list,ot_list


