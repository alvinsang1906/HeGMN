from utils.config import Parser
from model.Trainer import Trainer
from utils.utils import set_seed, get_config, nice_printer, load_data

if __name__ == "__main__":
    args = Parser().parse()
    config = get_config(args)
    set_seed(config['seed'])
    
    data = load_data(args.dataset)

    nice_printer(config)

    norm_ged = data.norm_ged_hetero if config['HGED'] is True else data.norm_ged_homoro
    trainer = Trainer(config, norm_ged)
    trainer.fit(data.train_data, data.test_data)
    trainer.score(data.train_data, data.test_data)
