import torch
import os
import copy

import warnings
warnings.filterwarnings("ignore")

from Utils.utils import set_seed, parse_cmd_line_params
from Utils.evaluation_metrics import print_evaluation_metrics
import unlearners

def main():
    args = parse_cmd_line_params()

    output_dir = f"results/{args.dataset}/{args.model_name_or_path.split('/')[-1]}"
    unlearner_name = args.unlearner if args.unlearner != "None" else "full"
    unlearner_name += "" if args.use_bad_teaching else "_light"
    unlearner_name += "_saliency" if args.saliency_map else ""
    output_dir = f"{output_dir}/{unlearner_name}_{args.lr}"
    output_dir = output_dir + f"_{args.epochs}/" if args.epochs > 1 else output_dir + "/"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Experiment {output_dir} already exists. Skipping...")
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Output Directory: ", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print("Seed: ", args.seed)
    set_seed(args.seed)

    retain_dataset = None # TODO: Load or create the retain dataset
    forget_dataset = None # TODO: Load or create the forget dataset
    val_dataset = None # TODO: Load or create the validation dataset
    test_dataset = None # TODO: Load or create the test dataset
    model = None # TODO: Load or train the Original model

    time = 0

    retain_dataloader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.batch, shuffle=True)
    forget_dataloader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch, shuffle=True)

    # switch unlearners 
    if args.unlearner != "None":
        unlearner_name = args.unlearner
        model.to(device)
        switcher = {
            "finetune": unlearners.finetune,
            "cfk": unlearners.cf_k,
            "neggrad": unlearners.neggrad,
            "advancedneggrad": unlearners.advancedneggrad,
            "unsir": unlearners.unsir,
            "scrub": unlearners.scrub,
            "bad_teaching": unlearners.bad_teaching
        }
        unlearner = switcher.get(unlearner_name, None)
        if unlearner is None:
            print("Invalid unlearner")
            return

        if unlearner_name == "unsir":
            time = unlearner(model, retain_dataloader, forget_dataloader, device, batch_size=args.batch, lr=args.lr, seed=args.seed)
        elif unlearner_name == "cfk":
            time = unlearner(model, retain_dataloader, forget_dataloader, device, lr=args.lr, unfreezed_encoder_layer=args.unfreeze_encoder_layer, seed=args.seed)
        elif unlearner_name == "bad_teaching":
            good_teacher = copy.deepcopy(model)
            if args.use_bad_teaching:
                print("Using bad teacher")
                bad_teacher = None # TODO: Load or create a bad teacher model
                bad_teacher.to(device)
            else:
                print("Not using bad teacher")
                bad_teacher = None
                unlearner_name = "bad_teaching_light"
            good_teacher.to(device)
            time = unlearner(model, bad_teacher, good_teacher, retain_dataset, forget_dataset, device, batch_size=args.batch, lr=args.lr, seed=args.seed)
        else:
            time = unlearner(model, retain_dataloader, forget_dataloader, device, lr=args.lr, seed=args.seed, num_epochs=args.epochs)

    output_dir = f"{output_dir}/{unlearner_name}/"
    os.makedirs(output_dir, exist_ok=True)

    dict = print_evaluation_metrics(model, forget_dataset, val_dataset, test_dataset, output_dir, device, save=args.save)

    with open(f"{output_dir}/evaluation_metrics.txt", 'w') as f:
        for key, value in dict.items():
            f.write(f'{key}: {value}\n')
        f.write(f'Unlearning Time: {time}\n')

    return


if __name__ == "__main__":
    main()