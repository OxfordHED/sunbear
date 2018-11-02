midx = slice(1,-1,None)
lidx = slice(None,-2,None)
hidx = slice(2,None,None)

def get_idx(ndim, axis=None, axis_idx=None):
    s = [midx] * ndim
    if axis is not None:
        if hasattr(axis, "__iter__"):
            for ax, axidx in zip(axis, axis_idx):
                s[ax] = axidx
        else:
            s[axis] = axis_idx
    return tuple(s)
