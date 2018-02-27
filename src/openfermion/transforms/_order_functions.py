




def up_then_down(mode_idx, num_modes):
    halfway = num_modes / 2
    if mode_idx % 2 == 0:
        return mode_idx // 2
    else:
        return mode_idx // 2 + halfway