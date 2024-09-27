import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna


class Order:
    def __init__(self, trade_type, entry_price, time, shares_count, stop_loss_level, take_profit_level):
        self.trade_type = trade_type
        self.entry_price = entry_price
        self.time = time
        self.shares_count = shares_count
        self.stop_loss_level = stop_loss_level
        self.take_profit_level = take_profit_level
        self.is_closed = False


class Strategy:
    def __init__(self, dataset_name):
        self.market_data = None
        self.trades = []
        self.capital = 1_000_000
        self.commission_rate = 0.00125
        self.portfolio_value_history = [1_000_000]
        self.shares_to_trade = 10
        self.dataset_name = dataset_name
        self.data_files = {
            "5m_train": "aapl_5m_train.csv",
            "5m_test": "aapl_5m_test.csv",
            "btc_train": "btc_project_train.csv",
            "btc_test": "btc_project_test.csv"
        }
        self.load_data(self.dataset_name)
        self.indicators_settings = {}
        self.active_indicators = []
        self.compute_indicators()
        self.set_signals()
        self.process_signals()
        self.optimal_combination = None
        self.highest_value = 0

    def load_data(self, timeframe):
        data_file = self.data_files.get(timeframe)
        if not data_file:
            raise ValueError("Invalid timeframe.")
        self.market_data = pd.read_csv(data_file)
        self.market_data.dropna(inplace=True)

    def compute_indicators(self):
        rsi_calc = ta.momentum.RSIIndicator(close=self.market_data['Close'], window=14)
        self.market_data['RSI'] = rsi_calc.rsi()

        short_sma = ta.trend.SMAIndicator(self.market_data['Close'], window=5)
        long_sma = ta.trend.SMAIndicator(self.market_data['Close'], window=21)
        self.market_data['SMA_short'] = short_sma.sma_indicator()
        self.market_data['SMA_long'] = long_sma.sma_indicator()

        macd_calc = ta.trend.MACD(close=self.market_data['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.market_data['MACD'] = macd_calc.macd()
        self.market_data['Signal_Line'] = macd_calc.macd_signal()

        stoch_calc = ta.momentum.StochasticOscillator(high=self.market_data['High'], low=self.market_data['Low'],
                                                      close=self.market_data['Close'], window=14, smooth_window=3)
        self.market_data['Stoch_%K'] = stoch_calc.stoch()
        self.market_data['Stoch_%D'] = stoch_calc.stoch_signal()

        bbands_calc = ta.volatility.BollingerBands(close=self.market_data['Close'], window=20, window_dev=2)
        self.market_data['Low_BB'] = bbands_calc.bollinger_lband()
        self.market_data['High_BB'] = bbands_calc.bollinger_hband()
        self.market_data['Mid_BB'] = bbands_calc.bollinger_mavg()

        self.market_data.dropna(inplace=True)
        self.market_data.reset_index(drop=True, inplace=True)

    def set_signals(self):
        self.indicators_settings = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal},
            'Bollinger': {'buy': self.bbands_buy_signal, 'sell': self.bbands_sell_signal}
        }

    def bbands_buy_signal(self, row, prev_row=None):
        return row['Close'] < row['Low_BB']

    def bbands_sell_signal(self, row, prev_row=None):
        return row['Close'] > row['High_BB']

    def stoch_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['Stoch_%K'] < prev_row['Stoch_%D'] and row['Stoch_%K'] > row[
            'Stoch_%D'] and row['Stoch_%K'] < 20

    def stoch_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['Stoch_%K'] > prev_row['Stoch_%D'] and row['Stoch_%K'] < row[
            'Stoch_%D'] and row['Stoch_%K'] > 80

    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70

    def sma_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['SMA_long'] > prev_row['SMA_short'] and row['SMA_long'] < row[
            'SMA_short']

    def sma_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['SMA_long'] < prev_row['SMA_short'] and row['SMA_long'] > row[
            'SMA_short']

    def macd_buy_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
        return False

    def macd_sell_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
        return False

    def process_signals(self):
        for indicator in list(self.indicators_settings.keys()):
            self.market_data[indicator + '_buy_signal'] = self.market_data.apply(
                lambda row: self.indicators_settings[indicator]['buy'](row, self.market_data.iloc[
                    row.name - 1] if row.name > 0 else None), axis=1)
            self.market_data[indicator + '_sell_signal'] = self.market_data.apply(
                lambda row: self.indicators_settings[indicator]['sell'](row, self.market_data.iloc[
                    row.name - 1] if row.name > 0 else None), axis=1)

        for indicator in list(self.indicators_settings.keys()):
            self.market_data[indicator + '_buy_signal'] = self.market_data[indicator + '_buy_signal'].astype(int)
            self.market_data[indicator + '_sell_signal'] = self.market_data[indicator + '_sell_signal'].astype(int)

        buy_signals = [indicator + '_buy_signal' for indicator in self.indicators_settings]
        sell_signals = [indicator + '_sell_signal' for indicator in self.indicators_settings]

        self.market_data['total_buy_signals'] = self.market_data[buy_signals].sum(axis=1)
        self.market_data['total_sell_signals'] = self.market_data[sell_signals].sum(axis=1)

    def execute_trades(self, best=False):
        active_indicators = self.optimal_combination if best else self.active_indicators
        total_active_indicators = len(active_indicators)

        for i, row in self.market_data.iterrows():
            if total_active_indicators <= 2:
                if self.market_data.total_buy_signals.iloc[i] == total_active_indicators:
                    self.open_trade('long', row)
                elif self.market_data.total_sell_signals.iloc[i] == total_active_indicators:
                    self.open_trade('short', row)
            else:
                if self.market_data.total_buy_signals.iloc[i] > (total_active_indicators / 2):
                    self.open_trade('long', row)
                elif self.market_data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                    self.open_trade('short', row)

            self.check_for_closures(row)

            portfolio_value = self.capital + sum(
                self.calculate_trade_value(trade, row['Close']) for trade in self.trades if not trade.is_closed)
            self.portfolio_value_history.append(portfolio_value)

    def open_trade(self, trade_type, row):
        stop_loss_level = row['Close'] * 0.95 if trade_type == 'long' else row['Close'] * 1.05
        take_profit_level = row['Close'] * 1.05 if trade_type == 'long' else row['Close'] * 0.95

        self.trades.append(
            Order(trade_type, row['Close'], row.name, self.shares_to_trade, stop_loss_level, take_profit_level))

        if trade_type == 'long':
            self.capital -= row['Close'] * self.shares_to_trade * (1 + self.commission_rate)
        else:
            self.capital += row['Close'] * self.shares_to_trade * (1 - self.commission_rate)

    def check_for_closures(self, row):
        for trade in self.trades:
            if not trade.is_closed and (
                    (trade.trade_type == 'long' and (
                            row['Close'] >= trade.take_profit_level or row['Close'] <= trade.stop_loss_level)) or
                    (trade.trade_type == 'short' and (
                            row['Close'] <= trade.take_profit_level or row['Close'] >= trade.stop_loss_level))):
                if trade.trade_type == 'long':
                    self.capital += row['Close'] * trade.shares_count * (1 - self.commission_rate)
                else:
                    self.capital -= row['Close'] * trade.shares_count * (1 + self.commission_rate)
                trade.is_closed = True

    def calculate_trade_value(self, trade, current_price):
        if trade.trade_type == 'long':
            return (current_price - trade.entry_price) * trade.shares_count if not trade.is_closed else 0
        else:
            return (trade.entry_price - current_price) * trade.shares_count if not trade.is_closed else 0

    def plot_results(self, use_best_combination=False):
        self.reset_strategy()
        if use_best_combination:
            self.execute_trades(best=True)
        else:
            self.execute_trades()

        plt.figure(figsize=(12, 8))
        plt.plot(self.portfolio_value_history)
        plt.title('Strategy Performance Over Time')
        plt.xlabel('Number of Trades')
        plt.ylabel('Portfolio Value')
        plt.show()

    def reset_strategy(self):
        self.trades.clear()
        self.capital = 1_000_000
        self.portfolio_value_history = [self.capital]

    def run_combinations(self):
        all_indicators = list(self.indicators_settings.keys())
        for r in range(1, len(all_indicators) + 1):
            for combo in combinations(all_indicators, r):
                self.active_indicators = list(combo)
                print(f"Running with indicators: {self.active_indicators}")
                self.execute_trades()

                final_value = self.portfolio_value_history[-1]
                if final_value > self.highest_value:
                    self.highest_value = final_value
                    self.optimal_combination = self.active_indicators.copy()

                self.reset_strategy()

        print(f"Best indicator combination: {self.optimal_combination} with a final value of: {self.highest_value}")

    def optimize_parameters(self):
        def objective(trial):
            self.reset_strategy()

            # Ajustar parámetros de los indicadores basados en las sugerencias de Optuna
            for indicator in self.optimal_combination:
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 5, 30)
                    self.adjust_rsi(rsi_window)
                elif indicator == 'SMA':
                    short_ma_window = trial.suggest_int('short_ma_window', 5, 20)
                    long_ma_window = trial.suggest_int('long_ma_window', 21, 50)
                    self.adjust_sma(short_ma_window, long_ma_window)
                elif indicator == 'MACD':
                    macd_fast = trial.suggest_int('macd_fast', 10, 20)
                    macd_slow = trial.suggest_int('macd_slow', 21, 40)
                    macd_sign = trial.suggest_int('macd_sign', 5, 15)
                    self.adjust_macd(macd_fast, macd_slow, macd_sign)
                elif indicator == 'Bollinger':
                    bb_window = trial.suggest_int('bb_window', 10, 50)
                    bb_dev = trial.suggest_float('bb_dev', 1.0, 3.0)
                    self.adjust_bbands(bb_window, bb_dev)
                elif indicator == 'Stoch':
                    stoch_k_window = trial.suggest_int('stoch_k_window', 5, 21)
                    stoch_d_window = trial.suggest_int('stoch_d_window', 3, 14)
                    stoch_smoothing = trial.suggest_int('stoch_smoothing', 3, 14)
                    self.adjust_stoch(stoch_k_window, stoch_d_window, stoch_smoothing)

            # Ejecutar la estrategia con los nuevos parámetros
            self.evaluate_signals()
            self.execute_trades(best=True)

            # Calcular el Sharpe Ratio
            sharpe_ratio = self.calculate_sharpe_ratio()

            # Optuna intentará maximizar este valor
            return sharpe_ratio

        # Crear el estudio de Optuna para maximizar el Sharpe Ratio
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)  # Ajusta n_trials según sea necesario

        # Asignar los mejores parámetros encontrados a la estrategia
        for indicator in self.optimal_combination:
            if indicator == 'RSI':
                self.adjust_rsi(study.best_params['rsi_window'])
            elif indicator == 'SMA':
                self.adjust_sma(study.best_params['short_ma_window'], study.best_params['long_ma_window'])
            elif indicator == 'MACD':
                self.adjust_macd(study.best_params['macd_fast'], study.best_params['macd_slow'],
                                 study.best_params['macd_sign'])
            elif indicator == 'Bollinger':
                self.adjust_bbands(study.best_params['bb_window'], study.best_params['bb_dev'])
            elif indicator == 'Stoch':
                self.adjust_stoch(study.best_params['stoch_k_window'], study.best_params['stoch_d_window'],
                                  study.best_params['stoch_smoothing'])

        print(f"Mejores parámetros encontrados: {study.best_params}")

    def calculate_sharpe_ratio(self, rf=0.0379):
        portfolio_returns = np.diff(np.log(self.portfolio_value_history))
        avg_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        if std_return == 0:  # Evitar la división por cero
            return 0

        sharpe_ratio = (avg_return - rf) / std_return
        return sharpe_ratio

    def adjust_rsi(self, window):
        rsi_calc = ta.momentum.RSIIndicator(close=self.market_data['Close'], window=window)
        self.market_data['RSI'] = rsi_calc.rsi()

    def adjust_sma(self, short_window, long_window):
        short_sma = ta.trend.SMAIndicator(self.market_data['Close'], window=short_window)
        long_sma = ta.trend.SMAIndicator(self.market_data['Close'], window=long_window)
        self.market_data['SMA_short'] = short_sma.sma_indicator()
        self.market_data['SMA_long'] = long_sma.sma_indicator()

    def adjust_macd(self, fast, slow, signal):
        macd_calc = ta.trend.MACD(close=self.market_data['Close'], window_slow=slow, window_fast=fast,
                                  window_sign=signal)
        self.market_data['MACD'] = macd_calc.macd()
        self.market_data['Signal_Line'] = macd_calc.macd_signal()

    def adjust_stoch(self, k_window, d_window, smoothing):
        stoch_calc = ta.momentum.StochasticOscillator(high=self.market_data['High'], low=self.market_data['Low'],
                                                      close=self.market_data['Close'], window=k_window,
                                                      smooth_window=d_window)
        self.market_data['Stoch_%K'] = stoch_calc.stoch()
        self.market_data['Stoch_%D'] = stoch_calc.stoch_signal().rolling(window=smoothing).mean()

    def adjust_bbands(self, window, deviation):
        bbands_calc = ta.volatility.BollingerBands(close=self.market_data['Close'], window=window, window_dev=deviation)
        self.market_data['Low_BB'] = bbands_calc.bollinger_lband()
        self.market_data['High_BB'] = bbands_calc.bollinger_hband()
        self.market_data['Mid_BB'] = bbands_calc.bollinger_mavg()

    def run_best_combination(self):

        if self.optimal_combination is None:
            print("No hay combinación óptima seleccionada.")
            return

        # Reiniciar la estrategia y ejecutar la mejor combinación
        self.reset_strategy()
        self.active_indicators = self.optimal_combination
        self.execute_trades(best=True)

        # Calcular métricas después de ejecutar la estrategia
        sharpe = self.calculate_sharpe_ratio()
        max_dd = self.calculate_max_drawdown()
        win_loss = self.calculate_win_loss_ratio()

        print(f"Mejor combinación de indicadores: {self.optimal_combination}")
        print(f"Sharpe Ratio: {sharpe}")
        print(f"Max Drawdown: {max_dd}")
        print(f"Win-Loss Ratio: {win_loss}")
        print(f"Valor final del portafolio: {self.portfolio_value_history[-1]}")

        # Graficar el rendimiento del portafolio
        self.plot_portfolio_performance()

        # Graficar precio del test y ubicación de las órdenes
        self.plot_trades_on_price()

    def calculate_sharpe_ratio(self, rf=0.0379):
        portfolio_returns = np.diff(np.log(self.portfolio_value_history))
        avg_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        sharpe_ratio = (avg_return - rf) / std_return
        return sharpe_ratio

    def calculate_max_drawdown(self):
        cumulative_max = np.maximum.accumulate(self.portfolio_value_history)
        drawdowns = (self.portfolio_value_history - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdowns)
        return max_drawdown

    def calculate_win_loss_ratio(self):
        wins = 0
        losses = 0
        for trade in self.trades:
            if trade.is_closed:
                profit_loss = (
                            trade.entry_price - self.portfolio_value_history[-1]) if trade.trade_type == 'long' else (
                            self.portfolio_value_history[-1] - trade.entry_price)
                if profit_loss > 0:
                    wins += 1
                else:
                    losses += 1

        if losses == 0:
            return float('inf')
        return wins / losses

    def plot_portfolio_performance(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_value_history, label="Portfolio Value")
        plt.title("Evolución del valor del portafolio")
        plt.xlabel("Número de operaciones")
        plt.ylabel("Valor del portafolio")
        plt.legend()
        plt.show()

    def plot_trades_on_price(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.market_data['Close'], label='Precio de Cierre')

        # Ubicar las órdenes
        for trade in self.trades:
            if trade.is_closed:
                if trade.trade_type == 'long':
                    plt.scatter(trade.time, trade.entry_price, color='green', marker='^', label="Compra (Long)")
                elif trade.trade_type == 'short':
                    plt.scatter(trade.time, trade.entry_price, color='red', marker='v', label="Venta (Short)")

        plt.title("Movimientos de precio y ubicación de las órdenes")
        plt.xlabel("Número de barras")
        plt.ylabel("Precio de cierre")
        plt.legend(loc="best")
        plt.show()