# -*- coding: utf-8 -*-
def num2info(num, delay=7, num_filter=3):
    """
    return tuple (chan, f, t)
    """
    block = delay * num_filter
    chan = num / block
    f = (num % block) / delay
    t = (num % block) % delay
    return (chan, f, t)
