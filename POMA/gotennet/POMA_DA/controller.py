import numpy as np

class AdaptiveDAWeightController:
    def __init__(self, init_w_mol=6e-7, init_w_sub=8e-8):
        self.w_mol = init_w_mol
        self.w_sub = init_w_sub
        self.w_reg = 1.0 
        self.ema_mol = None
        self.sub_energy_ratio = 0.3 
        self.max_w_mol = 4.0e-6  
        self.max_w_sub = 3e-7
        self.reg_threshold = 0.089

    def update(self, epoch, avg_reg, avg_mol_raw, avg_sub_raw):
        if avg_reg < self.reg_threshold:
            self.w_reg = max(0.2, self.w_reg * 0.98)
        else:
            self.w_reg = min(1.0, self.w_reg * 1.02)

        if avg_mol_raw < 1e-6: 
            return self.w_mol, self.w_sub, self.w_reg

        if self.ema_mol is None: 
            self.ema_mol = avg_mol_raw
        else:
            if avg_mol_raw > 1000 and avg_mol_raw > self.ema_mol * 15:
                return self.w_mol, self.w_sub, self.w_reg
            self.ema_mol = 0.8 * self.ema_mol + 0.2 * avg_mol_raw

        if avg_mol_raw < 5.0: 
            self.w_mol *= 1.03 
        
        return self.w_mol, self.w_sub, self.w_reg