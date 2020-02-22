import os
import torch
from modules import AudioModel, VisionModel


def audio_model(args):
    strides = [5, 4, 2, 2, 2]
    filter_sizes = [10, 8, 4, 4, 4]
    padding = [2, 2, 2, 2, 1]
    genc_hidden = 512
    gar_hidden = 256

    model = AudioModel(
        args,
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )
    return model

def vision_model(args):
    model = VisionModel(args)
    return model


def load_model(args):

    if args.experiment == "audio":
        model = audio_model(args)
    else:
        model = vision_model(args)
        # raise NotImplementedError

    # reload model
    if args.start_epoch > 0:
        print("### RELOADING MODEL FROM CHECKPOINT {} ###".format(args.start_epoch))
        model_fp = os.path.join(args.model_path, 'checkpoint_{}.tar'.format(args.start_epoch))
        model.load_state_dict(torch.load(model_fp))

    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate) 

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
            )

        print("### USING FP16 ###")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    args.num_gpu = torch.cuda.device_count()
    print("Using {} GPUs".format(args.num_gpu))

    model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    args.batch_size = args.batch_size * args.num_gpu
    return model, optimizer

def save_model(args, model, optimizer, best=False):
    if best:
        out = os.path.join(args.out_dir, 'best_checkpoint.tar')
    else:
        out = os.path.join(args.out_dir, 'checkpoint_{}.tar'.format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    torch.save(model.module.state_dict(), out)

    with open(os.path.join(args.out_dir, 'best_checkpoint.txt'), 'w') as f:
        f.write(str(args.current_epoch))