

import random

history = []

def get_delta(history: list[dict[str, int]]) -> float:
    return history[-1]["price"] - history[-2]["price"]

def make_decision(epoch: int, price: float):
    history.append({"epoch": epoch, "price": price})
    if (len(history) < 2):
        return {'Asset A':0.5, 'Cash': 0.5}
    if get_delta(history) > 0:
        return {'Asset A':0.7, 'Cash': 0.3}
    else:
        return {'Asset A':0.3, 'Cash': 0.7}