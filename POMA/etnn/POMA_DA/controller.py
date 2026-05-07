import numpy as np

class AdaptiveDAWeightController:
    def __init__(self, init_w_mol=3e-4, init_w_sub=4e-5):
        self.w_mol = init_w_mol
        self.w_sub = init_w_sub
        self.w_reg = 1.0 
        
        self.ema_mol = None
        self.sub_energy_ratio = 0.3 # 稍微下调，减少局部抖动
        self.max_w_mol = 1.0e-3     # ✨ 从 2e-3 降到 1e-3，适应 ETNN 较大的 molDA 数值
        self.max_w_sub = 2.0e-4     # ✨ 相应调低子结构上限
        
        # ✨ 阈值维持 0.15，这是我们通过监督学习日志确定的 ETNN 甜点位
        self.reg_threshold = 0.15 

    def update(self, epoch, avg_reg, avg_mol_raw, avg_sub_raw):
        # 1. 回归权重动态调整 (回归主权保护)
        if avg_reg < self.reg_threshold:
            # ✨ 修改点：底线从 0.2 提升到 0.5，衰减速度更平滑
            self.w_reg = max(0.5, self.w_reg * 0.99) 
        else:
            # 如果回归任务变差，迅速拉回权重
            self.w_reg = min(1.0, self.w_reg * 1.02)

        # 2. 异常值过滤
        if avg_mol_raw < 1e-6: 
            return self.w_mol, self.w_sub, self.w_reg

        if self.ema_mol is None: 
            self.ema_mol = avg_mol_raw
        else:
            if avg_mol_raw > 2000 and avg_mol_raw > self.ema_mol * 15:
                return self.w_mol, self.w_sub, self.w_reg
            self.ema_mol = 0.8 * self.ema_mol + 0.2 * avg_mol_raw

        # 3. 分子级 DA 权重调节
        if avg_mol_raw < 10.0: 
            self.w_mol *= 1.03 
        elif avg_mol_raw > 40.0: 
            self.w_mol *= 0.96 # 对齐太难时，退避速度加快
        
        self.w_mol = np.clip(self.w_mol, 1e-4, self.max_w_mol)

        # 4. 子结构级 DA 权重调节
        if avg_sub_raw > 1e-6:
            ideal_w_sub = (self.w_mol * avg_mol_raw * self.sub_energy_ratio) / avg_sub_raw
            self.w_sub = 0.8 * self.w_sub + 0.2 * ideal_w_sub
            
        self.w_sub = np.clip(self.w_sub, 1e-6, self.max_w_sub)

        return self.w_mol, self.w_sub, self.w_reg