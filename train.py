import torch
from torch.utils.tensorboard import SummaryWriter

from net import Net
from dataset import DroneDataset
from torch.utils.data import DataLoader


learning_rate = 0.0002
epoch = 5
total_strain_step = 0

if __name__ == '__main__':
    dataset = DroneDataset('./test')
    net = Net().to('cuda')
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss().to('cuda')
    writer = SummaryWriter('logs')
    for epoch in range(epoch):
        for img, label in DataLoader(dataset, batch_size=8, shuffle=True):
            img = img.to('cuda')
            label = label.to('cuda')
            output = net(img)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_strain_step += 1
            writer.add_scalar('train', loss.item(), total_strain_step)



