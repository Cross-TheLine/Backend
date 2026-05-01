from model import BallTrackerNet
import torch
from datasets import trackNetDataset
from general import validate
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help='device to use for evaluation')
    parser.add_argument('--num_workers', type=int, default=0, help='number of dataloader workers')
    args = parser.parse_args()

    val_dataset = trackNetDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = BallTrackerNet()
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    val_loss, precision, recall, f1 = validate(model, val_loader, device, -1)






