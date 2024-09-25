import pandas as pd
from typing import List

def gen_signals(datos: pd.DataFrame, strategy: List[int], signal_type: str, *strat_args):
    signals = pd.DataFrame(index=datos.index)
    signals['signal'] = 0

    for i, use_indicator in enumerate(strategy):
        if use_indicator:
            indicator_name = f'indicator_{i}'

            if signal_type == "BUY":
                signals['signal'] += datos[indicator_name + '_buy_signal'].apply(lambda x: 1 if x else 0)
            elif signal_type == "SELL":
                signals['signal'] += datos[indicator_name + '_sell_signal'].apply(lambda x: -1 if x else 0)

    if signal_type == "BUY":
        return signals['signal'].apply(lambda x: 1 if x > 0 else 0)
    elif signal_type == "SELL":
        return signals['signal'].apply(lambda x: -1 if x < 0 else 0)
