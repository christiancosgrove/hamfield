import pong
from model import HamFieldModel, HamiltonianLoss, Decoder, Integrator, Hamiltonian, PredictiveHamModel, Encoder, SymIntegratorP
import numpy as np
from skvideo.io import vwrite
from skimage.io import imsave
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

TIMESTEP = 1e-2
class PongDataset(Dataset):
    def __init__(self, num_balls, num_frames, resolution):
        system = pong.PingPong(num_balls, 0.1)

        f = system.frames(num_frames, resolution, 0.02)

        vwrite('out/out2.gif', f.transpose((1, 2, 3, 0)))

        # Subsample windows
        data = []
        for i in range(0, num_frames - window_size):
            data.append(np.expand_dims(f[:, i:i+window_size, :, :], 0))
        self.dat = np.concatenate(data, axis=0).astype(np.float32) / 255.0
        print('generated data with shape ', self.dat.shape)

    def __len__(self):
        return self.dat.shape[0]

    def __getitem__(self, i):
        return self.dat[i]

window_size = 30

def try_cuda(module):
    if torch.cuda.is_available():
        return module.cuda()
    # else:
        # print('Warning: CUDA not enabled')
    return module

def predicted_dynamics(df, num_frames, image, encoder: Encoder, decoder: Decoder, hamiltonian: Hamiltonian):
    # model = try_cuda(PredictiveHamModel(df, hamiltonian))
    hamiltonian.eval()
    encoder.eval()
    decoder.eval()
    integrator_p = try_cuda(SymIntegratorP(df, True))
    integrator_q = try_cuda(SymIntegratorP(df, False))
    curr_state = encoder(image)

    frames = []
    for i in range(num_frames):
        # Get the current frame imagined by the model
        f = decoder(*curr_state).cpu().detach().numpy()
        frames.append(f[:, :,window_size // 2, :, :])

        # Perform a step of integration
        # curr_state = model(*curr_state, TIMESTEP)
        output = hamiltonian(*curr_state)
        curr_state = integrator_p(*output, TIMESTEP)
        output = hamiltonian(*curr_state)
        curr_state = integrator_q(*output, TIMESTEP)
        # curr_state = (curr_state[0].detach(), curr_state[1].detach())

    hamiltonian.train()
    encoder.train()
    decoder.train()
    return np.transpose(np.squeeze(np.concatenate(frames, axis=0)), (0, 2, 3, 1))

def skvideo_write(name, arr):
    vwrite(name, (arr * 255).astype(np.uint8))

def train():

    epochs = 100
    df = 4

    model = try_cuda(HamFieldModel(df))
    integrator_p = try_cuda(SymIntegratorP(df, True))
    integrator_q = try_cuda(SymIntegratorP(df, False))

    print('generating data')
    
    dataset = PongDataset(4, 32*32*8, 32)
    mb_size = 1
    loader = DataLoader(dataset, mb_size, shuffle=True, num_workers=4)

    
    # hloss = try_cuda(HamiltonianLoss(df))

    optimizer = Adam(model.parameters(), lr=5e-4)

    for e in tqdm(range(epochs)):
        mloss = []
        for i, batch in tqdm(enumerate(loader)):
            batch = try_cuda(batch)
            optimizer.zero_grad()

            output, decoded = model(batch)
            _, p, _, _, q, _, _ = output
            # ham_loss = hloss(*output)
            decoder_loss = nn.MSELoss()(decoded, batch)

            frames = 20


            p_next_pred, q_next_pred = p, q
            loss = decoder_loss

            for f in range(frames):
                output = model.ham(p_next_pred, q_next_pred)
                p_next_pred, q_next_pred = integrator_p(*output, TIMESTEP)
                output = model.ham(p_next_pred, q_next_pred)
                p_next_pred, q_next_pred = integrator_q(*output, TIMESTEP)

                # p_next = p.detach()[:, :, f:, :, :]
                # q_next = q.detach()[:, :, f:, :, :]
                batch_sub = batch[:,:,f+1:, :, :]
                # time_size = p_next_pred.size(2)

                # p_next_pred = p_next_pred[:, :, :-f-1, :, :]
                # q_next_pred = q_next_pred[:, :, :-f-1, :, :]
                if True:#f == frames - 1:
                    loss += nn.MSELoss()(model.decoder(p_next_pred[:, :, :-f-1, :, :], q_next_pred[:, :, :-f-1, :, :]), batch_sub)
                    # loss += nn.MSELoss()(p_next_pred[:, :, :-f-1, :, :], p[:, :, f+1:, :, :].detach())
                    # loss += nn.MSELoss()(q_next_pred[:, :, :-f-1, :, :], q[:, :, f+1:, :, :].detach())
            # match_loss = nn.MSELoss()(p_next_pred, p_next) + nn.MSELoss()(q_next_pred, q_next)

            # loss = decoder_loss
            loss.backward()
            optimizer.step()
            mloss.append(loss.cpu().detach().numpy())


            def disp_tensor(t):
                im = t.cpu().detach().numpy()
                im = im[0, :, 0, :, :]
                im = np.transpose(im, (1, 2, 0))
                im *= 255
                im = im.astype(np.uint8)
                return im

            if i % 150 == 0:
                print('Mean loss: ', np.mean(mloss))
                imsave(f'out/out{e}.png', disp_tensor(decoded))
                # imsave(f'out/in{i}.png', disp_tensor(batch))

                # Get predicted dynamics and save to a file
                skvideo_write(f'out/test{e}.gif', predicted_dynamics(df, 48, batch[:1], model.encoder, model.decoder, model.ham))


if __name__ == '__main__':
    train()