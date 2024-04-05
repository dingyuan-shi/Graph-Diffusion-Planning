import argparse


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="configure all the settings")
    
    # device config
    parser.add_argument("-device", type=str, help="device, [cpu, cuda]", default="default")
    
    # path config 
    parser.add_argument("-path", type=str, help="data path", default="./sets_data")
    parser.add_argument("-model_path", type=str, help="model path", default="./sets_model")
    parser.add_argument("-res_path", type=str, help="results path", default="./sets_results")

    # data config
    parser.add_argument("-d_name", type=str, help="real data name chengdu, xian", default="")
    parser.add_argument("-n_vertex", type=int, help="number of vertices", default=20)
    parser.add_argument("-n_path", type=int, help="number of path", default=8000)
    parser.add_argument("-min_len", type=int, help="path length lower bound", default=4)
    parser.add_argument("-max_len", type=int, help="path length upper bound", default=15)
    
    # model config 
    parser.add_argument("-model_name", type=str, help="model name")
    parser.add_argument("-method", type=str, help="method: cold, naive")
    parser.add_argument("-plan", type=int, help="whether to plan, 1 plan, 0 no", default=0)
    parser.add_argument("-destroy", type=str, help="destroy manner: time, space")
    parser.add_argument("-beta_lb", type=float, help="beta lower bound", default=0.0001)
    parser.add_argument("-beta_ub", type=float, help="beta upper bound", default=0.1)
    parser.add_argument("-time_dim", type=int, help="dimension of pos emb", default=20)
    parser.add_argument("-max_T", type=int, help="max time step", default=10)
    parser.add_argument("-gmm_comp", type=int, help="gmm component number", default=4)
    parser.add_argument("-x_emb_dim", type=int, help="vertex embedding dim", default=50)
    parser.add_argument("-drop_cond", type=float, help="drop condition rate", default=0.3)
    parser.add_argument("-dims", type=str, help="temporal unet dims, finnaly multiply by n_groups", default="[100, 150, 200]")
    parser.add_argument("-hidden_dim", type=int, help="hidden dim for time and condition", default=20)
    parser.add_argument("-n_groups", type=int, help="number of groups for group normalization", default=8)
    
    # ngram model config 
    parser.add_argument("-n_gram", type=int, help="n gram", default=1)
    
    # hmm model config 
    parser.add_argument("-hidden_states", type=int, help="hidden states for hmm", default=10)
    
    # lstm model config 
    # hidden_size, num_layers
    parser.add_argument("-hidden_size", type=int, help="hidden size for lstm", default=30)
    parser.add_argument("-num_layers", type=int, help="num_layers for lstm", default=3)
    
    # training config
    parser.add_argument("-n_epoch", type=int, help="number of epoch", default=200)
    parser.add_argument("-bs", type=int, help="batch size", default=32)
    parser.add_argument("-lr", type=float, help="learning rate", default=0.001)
    parser.add_argument("-gmm_samples", type=int, help="gmm samples", default=3000)
    
    # eval config
    parser.add_argument("-eval_num", type=int, help="evaluation sample number, int", default=1000)
    
    return parser