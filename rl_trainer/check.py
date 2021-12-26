import torch  # 命令行是逐行立即执行的
import os
content_qmix = torch.load('./agent/90_qmix_net_params.pth')
content_rnn = torch.load('./agent/90_rnn_net_params.pth')
#print(content_qmix.keys())   # keys()
#print(content_rnn.keys())
# 之后有其他需求比如要看 key 为 model 的内容有啥
#print(content_qmix['model'])
for key in content_rnn.keys():
    print(content_rnn[key])
for key in content_qmix.keys():
    print(content_qmix[key])