#! /usr/bin/env python3

import csv
import json
import os
import sys

# Empêcher la création de __pycache__
sys.dont_write_bytecode = True


from scoring.scoring import get_local_score, show_result
import pandas as pd
from bot_trade import make_decision as decision_generator
import matplotlib.pyplot as plt


def find_csv_file(path_csv: str) -> pd.DataFrame:
    if not os.path.exists(path_csv):
        raise FileNotFoundError(f"Le fichier CSV {path_csv} n'existe pas")
    prices_list = [
        pd.read_csv(path_csv, index_col=0)
    ]
    prices = pd.concat(prices_list, axis=1)
    prices["Cash"] = 1
    return prices

def validate_decision(decision: dict) -> bool:
    expected_keys = {'Asset A', 'Cash'}
    if set(decision.keys()) != expected_keys:
        print(f"ERREUR: Les clés attendues sont {expected_keys}, mais reçu {set(decision.keys())}")
        return False

    for key, value in decision.items():
        if not isinstance(value, (int, float)):
            print(f"ERREUR: La valeur pour '{key}' n'est pas numérique: {value}")
            return False
        if value < 0 or value > 1:
            print(f"ERREUR: La valeur pour '{key}' doit être entre 0 et 1, reçu: {value}")
            return False
    
    total = sum(decision.values())
    if abs(total - 1.0) > 0.00001:
        print(f"ERREUR: La somme des allocations doit être égale à 1, mais vaut {total}")
        return False
    
    return True

def main():
    max_epoch = 100
    global_pnl = 0
    global_base_score = 0
    if (len(sys.argv) > 1):
        path_csv = sys.argv[1]
        prices = find_csv_file(path_csv=path_csv)
    else:
        raise ValueError("No path to the csv file provided, ./main.py <path_to_csv>")
    for i in range(0, max_epoch):
        output = []
        for index, row in prices.iterrows():
            decision = decision_generator(int(index), float(row['Asset A']))
            if not validate_decision(decision):
                raise ValueError(f"Décision invalide: {decision}")
            decision['epoch'] = int(index)
            output.append(decision)
        positions = pd.DataFrame(output).set_index("epoch")
        local_score = get_local_score(prices=prices, positions=positions)
        global_base_score += local_score["scores"]["base_score"]
        global_pnl += local_score["stats"]["cumulative_return"]*100
        print(f"Run {i+1}/{max_epoch} - PnL: {local_score['stats']['cumulative_return']*100}%, Base Score: {local_score["scores"]["base_score"]}")
        # if len(sys.argv) > 2 and sys.argv[2] == "--show-graph":
        #     show_result(local_score, is_show_graph=True)
        # else:
        #     show_result(local_score, is_show_graph=False)
        #     print("\033[91mpour afficher le graphique, utilisez la commande --show-graph: ./main.py <path_to_csv> --show-graph\033[0m")
    print(f"Score moyen sur {max_epoch} runs: {global_pnl/max_epoch}")
    print(f"Base score moyen sur {max_epoch} runs: {global_base_score/max_epoch}")

if __name__ == "__main__":
    main()