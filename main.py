from os.path import join
import torch
from loader.gen_graph import DataGenerator
from loader.dataset import TrajFastDataset
from utils.argparser import get_argparser
from utils.evaluate import Evaluator


if __name__ == "__main__":
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    parser = get_argparser()
    args = parser.parse_args()
    
    # set device
    if args.device == "default":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(device)
    
    # set dataset
    if args.d_name == "":
        n_vertex = args.n_vertex
        name = f"v{args.n_vertex}_p{args.n_path}_{args.min_len}{args.max_len}"
        dataset = DataGenerator(args.n_vertex, args.n_path, args.min_len, args.max_len, device, args.path, name)
    elif args.d_name != "":
        date = "20161101" if args.d_name == "chengdu" else "20161001"
        dataset = TrajFastDataset(args.d_name, [date], args.path, device, is_pretrain=True)
        n_vertex = dataset.n_vertex
        print(f"vertex: {n_vertex}")
        
    # before train, record the info
    with open(join(args.model_path, f"{args.model_name}.info"), "w") as f:
        f.writelines(str(args))
    
    # set model
    if args.method == "seq":
        from models_seq.seq_models import Destroyer, Restorer
        from models_seq.eps_models import EPSM
        from models_seq.trainer import Trainer
        
        suffix = "cd" if args.d_name == "chengdu" else "xa"
        
        betas = torch.linspace(args.beta_lb, args.beta_ub, args.max_T)
        destroyer = Destroyer(dataset.A, betas, args.max_T, device)
        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
        dims = eval(args.dims)
        eps_model = EPSM(dataset.n_vertex, x_emb_dim=args.x_emb_dim, dims=dims, device=device, hidden_dim=args.hidden_dim, pretrain_path=pretrain_path)
        model = Restorer(eps_model, destroyer, device)
        
        trainer = Trainer(model, dataset, args.model_path)
        trainer.train_gmm(gmm_samples=args.gmm_samples, n_comp=args.gmm_comp)
        trainer.train(args.n_epoch, args.bs, args.lr)
        model.eval()    
        torch.save(model, join(args.model_path, f"{args.model_name}.pth"))
        
        model.eval()    
        
    elif args.method == "plan":
        from planner.planner import Planner
        from planner.trainer import Trainer
        suffix = "cd" if args.d_name == "chengdu" else "xa"

        pretrain_path = join(args.path, f"{args.d_name}_node2vec.pkl")
        restorer = torch.load(f"./sets_model/no_plan_gen_{suffix}.pth")
        destroyer = restorer.destroyer
        model = Planner(dataset.G, dataset.A, restorer, destroyer, device, x_emb_dim=args.x_emb_dim, pretrain_path=pretrain_path)
        trainer = Trainer(model, dataset, device, args.model_path)
        trainer.train(args.n_epoch, args.bs, args.lr)
        model.eval()
        torch.save(model, join(args.model_path, f"{args.model_name}.pth"))
        
        
    if args.method != "plan":
        gen_paths = model.sample(args.eval_num)
        real_paths = dataset.get_real_paths(args.eval_num)
        
        evaluator = Evaluator(real_paths, gen_paths, model, n_vertex, name=join(args.res_path, f"{args.model_name}_pure_gen"))
        res = evaluator.eval_all()
        print(res)
        with open(join(args.res_path, f"{args.model_name}.res"), "w") as f:
            f.writelines(str(res))
            
    if args.method == "plan":
        from utils.evaluate_plan import Evaluator
        suffix = "cd" if args.d_name == "chengdu" else "xa"
        evaluator = Evaluator(model, dataset)
        evaluator.eval(args.eval_num, suffix)