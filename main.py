from loader.txt_processor import load_config
from attribution.SVattr import Attribution
from models import  DNAMTA, Nlinear, causalMTA, LR, SP, DCRMTA
from pprint import pprint
model_dict = {
        'SP': SP,
        'LR': LR,
        'Nlinear': Nlinear,
        'causalMTA': causalMTA,
        'ARNN': DNAMTA,
        'DCRMTA': DCRMTA
    }

cfgs = load_config('./configs/DCRMTA.txt')
data_cfg = load_config('./data/mock_cfg.txt')
# nohup python main.py 2>&1 | tee ./log/terminal_info/test.log
# ps -ef |grep tangjiaming  |awk '{print $2}'|xargs kill -9


if __name__ == '__main__':
    pprint('CONFIGS:\n')
    pprint(cfgs)
    pprint(data_cfg)
    trainer = model_dict[cfgs['model_name']].Trainer(cfgs, data_cfg)
    if not cfgs['pretrained']:
        trainer.train_eval()
    Attribution(cfgs, data_cfg, trainer)
      
