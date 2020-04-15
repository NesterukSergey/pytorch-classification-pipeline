try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise RuntimeError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda)."
        )
        

def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
#     data_loader_iter = iter(data_loader)
#     x, y = next(data_loader_iter)
#     try:
#         writer.add_graph(model, x)
#     except Exception as e:
#         print("Failed to save model graph: {}".format(e))
    return writer

