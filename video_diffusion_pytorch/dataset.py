import numpy as np
import os
import sys
sys.path.append("./tools")
sys.path.append("../tools")
import tools.uniio as uniio

import torch
from torch.utils.data import Dataset

from einops import rearrange

# Definitions
TOTAL_SIMULATION_TIME = 20 # length of sequences

def load_sim_file(data_path, sim, type_name, idx):
    uniPath = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # TOTAL_SIMULATION_TIME files per sim
    print(uniPath)
    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr

class MantaFlow2DSimSequenceDataset(Dataset):
    """Provides dt, vt, tau in a sequence of length TOTAL_SIMULATION_TIME"""
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.batched_densities = []
        self.batched_velocities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                # Note: fill in each batch with a sequence of len TOTAL_SIMULATION_TIME
                seq_of_density_fields = []
                seq_of_velocity_fields = []
                for i in range(0, TOTAL_SIMULATION_TIME):
                    seq_of_density_fields.append(load_sim_file(data_path, sim, 'density', i))
                    seq_of_velocity_fields.append(load_sim_file(data_path, sim, 'vel', i))

                seq_of_density_fields = np.reshape( seq_of_density_fields, (len(seq_of_density_fields), grid_height, grid_width, 1) )
                print("Read uni files (density), total data " + format(seq_of_density_fields.shape))

                seq_of_velocity_fields = np.reshape( seq_of_velocity_fields, (len(seq_of_velocity_fields), grid_height, grid_width, 3) )
                seq_of_velocity_fields = seq_of_velocity_fields[:, :, :, :2]
                print("Read uni files (velocity), total data " + format(seq_of_velocity_fields.shape))

                # Note: Batchify the sims
                self.batched_densities.append(seq_of_density_fields)
                self.batched_velocities.append(seq_of_velocity_fields)


        num_densities = len(self.batched_densities)
        num_velocities = len(self.batched_velocities)
        NUM_SIMS = 2
        assert num_densities >= NUM_SIMS, f"Number of simulations should be at least {NUM_SIMS}"
        assert self.batched_velocities[0].shape[0] == TOTAL_SIMULATION_TIME, f"Simulation time should be {TOTAL_SIMULATION_TIME}"
        print(f'Loaded {num_densities} densities')
        print(f'Loaded {num_velocities} densities')

        self.batched_densities = np.array(self.batched_densities)
        self.batched_velocities = np.array(self.batched_velocities)
        print(f"Shape of batchified densities sims: {self.batched_densities.shape}")
        print(f"Shape of batchified velocities sims: {self.batched_velocities.shape}") # [20, SIM_TIME, 64, 64, 2]

        # Note: Identical entries normalized by the TOTAL_SIMULATION_TIME
        tau_range = torch.arange(0, TOTAL_SIMULATION_TIME)
        XX, _ = torch.meshgrid(tau_range, tau_range, indexing="ij")
        self.tau_t = torch.stack([ torch.tile(XX[t], dims=(TOTAL_SIMULATION_TIME, 1))[: grid_height, : grid_width].unsqueeze(-1).permute(2, 0, 1) for t in range(TOTAL_SIMULATION_TIME) ], dim=0).float()
        self.tau_t /= (TOTAL_SIMULATION_TIME - 1) # [SIM_TIME, 1, 64, 64]
        print(f"Shape of tau: {self.tau_t.shape}")

        # Note: Transform at construction to save compute (in-memory)

        self.batched_densities_t = []
        self.batched_velocities_t = []
        for idx in range(self.__len__()):
            self.batched_densities[idx] = (self.batched_densities[idx] - self.batched_densities[idx].min()) / (self.batched_densities[idx].max() - self.batched_densities[idx].min())
            self.batched_velocities[idx] = (self.batched_velocities[idx] - self.batched_velocities[idx].min()) / (self.batched_velocities[idx].max() - self.batched_velocities[idx].min())
            d0_t = torch.from_numpy(self.batched_densities[idx].transpose(0, 3, 1, 2)).float() # hwc to chw
            v0_t = torch.from_numpy(self.batched_velocities[idx].transpose(0, 3, 1, 2)).float() # hwc to chw

            if self.transform_ops is not None:
                d0_t = self.transform_ops(d0_t)
                v0_t = self.transform_ops(v0_t)
            
            self.batched_densities_t.append(d0_t)
            self.batched_velocities_t.append(v0_t)
        
        self.batched_densities_t = torch.stack(self.batched_densities_t, dim=0).float()
        self.batched_velocities_t = torch.stack(self.batched_velocities_t, dim=0).float()
        print(f"Shape of torch batchified densities sims: {self.batched_densities_t.size()}")
        print(f"Shape of torch batchified velocities sims: {self.batched_velocities_t.size()}") # [20, 100, 64, 64, 2]


    def __getitem__(self, idx):
        d0_seq_t = self.batched_densities_t[idx]
        v0_seq_t = self.batched_velocities_t[idx]
        x = torch.concat([d0_seq_t, v0_seq_t], dim=1)
        x = rearrange(x, 'f c h w -> c f h w')
        return x#, self.tau_t

    def __len__(self):
        assert self.batched_densities.shape[0] == self.batched_velocities.shape[0]
        return self.batched_velocities.shape[0]
