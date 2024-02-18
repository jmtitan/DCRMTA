from loader.txt_processor import load_config
from attribution.SVattr import Attribution
from models import Nlinear, causalMTA, LR, DNAMTA, DCRMTA

model_dict = {
        'LR':LR,
        'Nlinear':Nlinear,
        'causalMTA':causalMTA,
        'DNAMTA':DNAMTA,
        'DCRMTA':DCRMTA
    }


data_cfg = load_config('./data/criteo_cfg.txt')

# nohup python attr.py 2>&1 | tee ./log/terminal_info/attr/test.log &
# ps -ef |grep tangjiaming  |awk '{print $2}'|xargs kill -9


if __name__ == '__main__':

    mn = ['LR', 'Nlinear','causalMTA','DNAMTA','DCRMTA']
    # mn = ['Nlinear']
    for i in mn:
        cfgs = load_config(f'./configs/{i}.txt')
        print(i)
        trainer = model_dict[cfgs['model_name']].Trainer(cfgs, data_cfg)
        if  cfgs['pretrained']:
            Attribution(cfgs, data_cfg, trainer=trainer)
        print('split label ==============================>')
