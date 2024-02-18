from loader.txt_processor import load_config
from models import  Nlinear, causalMTA, LR, SP, DCRMTA, DNAMTA
from pprint import pprint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model name", type=str)
model_dict = {
        'SP':SP,
        'LR':LR,
        'Nlinear':Nlinear,
        'causalMTA': causalMTA,
        'DNAMTA':DNAMTA,
        'DCRMTA':DCRMTA
    }


data_cfg = load_config('./data/criteo_cfg.txt')

# nohup python exp.py 2>&1 | tee ./log/terminal_info/criteo/test.log &
# nohup python exp.py 2>&1 | tee ./log/terminal_info/mock/test.log & 

# ps -ef |grep tangjiaming  |awk '{print $2}'|xargs kill -9


if __name__ == '__main__':
    args = parser.parse_args()
    pprint('CONFIGS:\n')
    # mn = ['LR','Nlinear','DNAMTA','CausalMTA','DCRMTA']
    # mn = ['Nlinear']
    # mn =['DNAMTA']
    # mn= ['causalMTA']
    # mn = ['DCRMTA']
    # for i in mn:
    model = args.model
    cfgs = load_config(f'./configs/{model}.txt')
    pprint(cfgs)
    pprint(data_cfg)
    print('Training --------------------------------->')
    trainer = model_dict[cfgs['model_name']].Trainer(cfgs, data_cfg)
    trainer.train_eval()
    print('split label ==============================>')