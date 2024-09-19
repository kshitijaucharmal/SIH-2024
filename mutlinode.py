import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class ColorizationDataset(Dataset):
    def __init__(self, pairs, split='train', size=256):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)
        
        self.split = split
        self.size = size
        self.pairs = pairs

    def __getitem__(self, idx):
        input_path, output_path = self.pairs[idx]
        
        input_img = Image.open(input_path)
        input_img = self.transforms(input_img)
        input_img = np.array(input_img)
        input_lab = transforms.ToTensor()(input_img)
        L = input_lab[[0], ...] / (torch.max(input_lab) - torch.min(input_lab)) - 1
        
        output_img = Image.open(output_path).convert("RGB")
        output_img = self.transforms(output_img)
        output_img = np.array(output_img)
        output_lab = rgb2lab(output_img).astype("float32")
        output_lab = transforms.ToTensor()(output_lab)
        ab = output_lab[[1, 2], ...] / (torch.max(output_lab) - torch.min(output_lab))
        
        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.pairs)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch, dataloader):
        b_sz = len(next(iter(dataloader))['L'])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(dataloader)}")
        dataloader.sampler.set_epoch(epoch)
        total_loss = 0
        for batch in dataloader:
            source = batch['L'].to(self.local_rank)
            targets = batch['ab'].to(self.local_rank)
            loss = self._run_batch(source, targets)
            total_loss += loss
        return total_loss / len(dataloader)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            train_loss = self._run_epoch(epoch, self.train_data)
            val_loss = self._run_epoch(epoch, self.val_data)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs(train_pairs, val_pairs, batch_size):
    train_dataset = ColorizationDataset(pairs=train_pairs, split='train')
    val_dataset = ColorizationDataset(pairs=val_pairs, split='val')

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_dataset, val_dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    base_dir = "./dataset/v2"
    categories = ['agri', 'barrenland', 'grassland', 'urban']
    input_output_pairs = []
    for category in categories:
        input_folder = os.path.join(base_dir, category, 's1')
        output_folder = os.path.join(base_dir, category, 's2')
        input_images = sorted(glob.glob(os.path.join(input_folder, "*.png")))
        output_images = sorted(glob.glob(os.path.join(output_folder, "*.png")))
        for input_img, output_img in zip(input_images, output_images):
            input_output_pairs.append((input_img, output_img))

    np.random.seed(123)
    input_output_pairs = np.random.permutation(input_output_pairs)
    split = int(0.8 * len(input_output_pairs))
    train_pairs = input_output_pairs[:split]
    val_pairs = input_output_pairs[split:]

    train_dataset, val_dataset, model, optimizer = load_train_objs(train_pairs, val_pairs, batch_size)
    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)
    
    trainer = Trainer(model, train_data, val_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Distributed training job for image colorization')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 16)')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to the base dataset directory')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size, args.base_dir)