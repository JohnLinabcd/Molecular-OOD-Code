import numpy as np

class AdaptiveDAWeightController:
    def __init__(self, init_w_mol=3e-4, init_w_sub=4e-5):
        self.w_mol = init_w_mol
        self.w_sub = init_w_sub
        # --- 回归损失权重初始化 ---
        # 理由：默认权重为1.0，后期根据Reg Loss表现进行衰减
        self.w_reg = 1.0 
        self.ema_mol = None
        self.sub_energy_ratio = 0.3 
        self.max_w_mol = 2.0e-3  
        self.max_w_sub = 1.5e-4
        # --- 设置回归权重衰减阈值 ---
        # 理由：根据日志，Reg在0.04附近是泛化性能的黄金点，低于此值说明开始过拟合源域
        self.reg_threshold = 0.04 

    def update(self, epoch, avg_reg, avg_mol_raw, avg_sub_raw):
        # --- 回归权重自适应调节 ---
        # 逻辑：如果回归损失已经低于阈值，说明源域学得太“死”了，开始按比例衰减其权重
        if avg_reg < self.reg_threshold:
            # 采用温和的乘法衰减，最低保留0.2的权重以维持基础预测能力
            self.w_reg = max(0.2, self.w_reg * 0.98)
        else:
            # 如果Reg回升（说明对齐干扰了预测），则缓慢恢复回归权重
            self.w_reg = min(1.0, self.w_reg * 1.02)

        if avg_mol_raw < 1e-6: 
            # 返回三个权重
            return self.w_mol, self.w_sub, self.w_reg

        if self.ema_mol is None: 
            self.ema_mol = avg_mol_raw
        else:
            if avg_mol_raw > 1000 and avg_mol_raw > self.ema_mol * 15:
                return self.w_mol, self.w_sub, self.w_reg
            self.ema_mol = 0.8 * self.ema_mol + 0.2 * avg_mol_raw

        # 分子级 DA 权重调节
        if avg_mol_raw < 5.0: 
            self.w_mol *= 1.03 
        elif avg_mol_raw > 20.0: 
            self.w_mol *= 0.97 
        self.w_mol = np.clip(self.w_mol, 1e-4, self.max_w_mol)

        # 子结构级 DA 权重调节
        if avg_mol_raw < 15.0:
            self.sub_energy_ratio = max(0.1, self.sub_energy_ratio * 0.98)
            
        if avg_sub_raw > 1e-6:
            # 这里的逻辑是动态平衡 Mol 和 Sub 的能量级
            ideal_w_sub = (self.w_mol * avg_mol_raw * self.sub_energy_ratio) / avg_sub_raw
            self.w_sub = 0.8 * self.w_sub + 0.2 * ideal_w_sub
            
        self.w_sub = np.clip(self.w_sub, 1e-6, self.max_w_sub)

        return self.w_mol, self.w_sub, self.w_reg