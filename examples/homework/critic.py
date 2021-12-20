import torch.nn as nn
import torch.nn.functional as F

'''
class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer

        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        return x'''

class Critic_CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layer=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layer = num_hidden_layer


        self.conv1 = nn.Sequential(  # input shape (1,6,8)
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=8,  # n_filter
                      kernel_size=2,  # filter size
                      stride=1,  # filter step
                      padding='same'  # con2d出来的图片大小不变
                      ),  # output shape (8,28,28)
            nn.MaxPool2d(kernel_size=2)  # 2x2采样，output shape (8,3,4)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 2, 1, 'same'),  # output shape (16,3,4)
                                   nn.ReLU())

        self.linear_in = nn.Linear(16 * 3 * 4, hidden_size)

        if self.num_hidden_layer > 0:
            hid_net = []
            for _ in range(self.num_hidden_layer):
                hid_net.append(nn.Linear(hidden_size, hidden_size))
                hid_net.append(nn.ReLU())
            self.linear_hid = nn.Sequential(*hid_net)

        self.linear_out = nn.Linear(hidden_size, self.output_size)


    def forward(self, x):
        #print(x,x.shape)
        x = self.conv1(x)
        #print(x,x.shape)
        x = self.conv2(x)
        #print(x, x.shape)
        x = x.view(x.size(0), -1)
        x = self.linear_in(x)
        if self.num_hidden_layer > 0:
            x = self.linear_hid(x)
        x = self.linear_out(x)
        return x

class Dueling_Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x2 = F.relu(self.linear1(x))

        # value
        y1 = self.linear2(x1)
        # advantage
        y2 = self.linear3(x2)
        # y2.mean(dim=1,keepdim=True)：dim=1按行取均值；keepdim=True时，输出与输入维度相同，仅仅时输出在求均值的维度上元素个数变为1。
        # keepdim=False时，输出比输入少一个维度，就是指定的dim求均值的维度。
        x3 = y1 + y2 - y2.mean(dim=1, keepdim=True)

        return x3

      
class openai_critic(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, 128)
        self.linear_c2 = nn.Linear(128, 64)
        self.linear_c = nn.Linear(64, 1)
        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

