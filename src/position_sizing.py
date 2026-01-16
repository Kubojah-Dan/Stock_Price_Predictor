def kelly_fraction(win_rate, win_loss_ratio):
    return max(0, win_rate - (1 - win_rate) / win_loss_ratio)

def volatility_targeting(vol, target_vol=0.01):
    return min(1.0, target_vol / vol)
