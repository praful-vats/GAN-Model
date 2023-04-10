from torch.utils.data import DataLoader
from dataset import MyDataset

import os
print(os.getcwd())

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, required=True, help='path to dataset')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
parser.add_argument('--model', type=str, default='cycle_gan', help='type of model to use (cycle_gan)')
parser.add_argument('--no_dropout', action='store_true', help='if specified, do not use dropout in the generator')
parser.add_argument('--gpu_ids', type=str, default='0', help='ids of GPUs to use, separated by comma (e.g. "0,1")')
args = parser.parse_args()



# set the root directory of your dataset
dataroot = '/content/drive/MyDrive/format'

# initialize the dataset
dataset = MyDataset(dataroot)

# initialize the dataloader
batch_size = 1
shuffle = True
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Train loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_dataloader):
        # Set model input
        # real_A = data[0].to(device)
        # real_B = data[1].to(device)
        real_A, real_B = data.to(device)


        # Train generators
        optimizer_G.zero_grad()

        # Identity loss
        same_A = G_BA(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0
        same_B = G_AB(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0

        # GAN loss
        fake_A = G_BA(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_AB = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))
        fake_B = G_AB(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_BA = criterion_GAN(pred_fake, torch.ones_like(pred_fake).to(device))

        # Cycle loss
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A) * 10.0
        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B) * 10.0

        # Total generator loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        optimizer_G.step()

        # Train discriminators
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
        pred_real = D_B(real_B)
        loss_D_real += criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
        loss_D_real /= 2.0

        # Fake loss
        fake_A = G_BA(real_B)
        pred_fake = D_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
        fake_B = G_AB(real_A)
        pred_fake = D_B(fake_B.detach())
        loss_D_fake += criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
        loss_D_fake /= 2.0

        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake

        # Backward and optimize
        loss_D.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        # Print losses
        if i == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')
        else:
            print(f'Step [{i+1}/{len(train_dataloader)}], Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')
