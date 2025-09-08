
import torch

BATCH_SIZE = 1
NUM_WORKER = 1
FILE_NAME = 'train_'

LEARN_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SET_NAME = 'Tay_Ji'
LABEL_RATE = 10
EXTRA_WORD = ''
PRE_EPOCH = 0
START_EPOCH = 0
END_EPOCH = 200
GAP_EPOCH = 5
Pior = 0.25
TEST_EPOCH_NUM = 'best'
sever_root = ''
data_root_path = r'D:\MasterProgram\paper2\GIS\L3'
test_path = r"D:\MasterProgram\paper2\GIS\L3\PreTraining"
base_root = r'D:\MasterProgram\paper2\MyNET'  # 数据以config中的文件位置为主
model_paths = r"D:\MasterProgram\paper2\MyNET\log\pre\20_net_g.pth"
last_path = None
sentinel_use_flag = True
data_enhancement_flag = True

if sentinel_use_flag:
    inchanels=51
else:
    inchanels=6

config = dict(
    dataset=dict(
        train=dict(
            train_flag=True,
            num_positive_train_samples=135,
            sub_minibatch=5,
            ccls=1,
            ratio=40,
            im_cmean=[558.80828094, 751.90450668, 490.21277618, 2193.66233826],
            im_cstd=[100.82819428, 148.36214528, 165.81017142, 393.28385579]
        ),
        test=dict(
            train_flag=False,
            num_positive_train_samples=135,
            sub_minibatch=5,
            ccls=1,
            ratio=40,
            im_cmean=[558.80828094, 751.90450668, 490.21277618, 2193.66233826],
            im_cstd=[100.82819428, 148.36214528, 165.81017142, 393.28385579]
        )
    ),
    model=dict(
        type='FreeOCNet',
        params=dict(
            in_channels=inchanels,  # 6 or 45
            num_classes=1,
            block_channels=(64, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )
    ),

    meta=dict(
        palette=[
            [0, 0, 0],
            [176, 48, 96]],
    ),
)
