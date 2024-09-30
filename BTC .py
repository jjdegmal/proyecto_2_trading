#!/usr/bin/env python
# coding: utf-8

# In[4]:


import itertools
import optuna
import pandas as pd
import ta  
import csv
class Position:
    def __init__(self, ticker, n_shares, price, timestamp, position_type):
        self.ticker = ticker
        self.n_shares = n_shares
        self.price = price
        self.timestamp = timestamp
        self.position_type = position_type

# Función para cargar los datos de entrenamiento y prueba
def cargar_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# Función para generar todas las combinaciones de indicadores técnicos
def generate_combinations(indicators):
    all_combinations = []
    for r in range(1, len(indicators) + 1):
        combinations = itertools.combinations(indicators, r)
        all_combinations.extend(combinations)
    return all_combinations

def backtest(data, strategy, sl, tp, n_shares, initial_capital, commission,
             rsi_window=None, rsi_lower=None, rsi_upper=None,
             bollinger_window=None, bollinger_stddev=None, 
             macd_fast=None, macd_slow=None, macd_signal=None,
             cci_window=None, stochastic_k=None, stochastic_d=None):

    # Calcular los indicadores fuera del bucle de backtest
    rsi = ta.momentum.RSIIndicator(close=data['Close'], window=rsi_window).rsi() if "RSI" in strategy else None
    bollinger = ta.volatility.BollingerBands(close=data['Close'], window=bollinger_window, window_dev=bollinger_stddev) if "Bollinger" in strategy else None
    macd = ta.trend.MACD(close=data['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal) if "MACD" in strategy else None
    cci = ta.trend.CCIIndicator(high=data['High'], low=data['Low'], close=data['Close'], window=cci_window).cci() if "CCI" in strategy else None
    stochastic = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], 
                                                  window=stochastic_k, smooth_window=stochastic_d) if "Stochastic" in strategy else None

    # Variables del capital y posición inicial
    capital = initial_capital
    position = 0  # Número de acciones en la posición
    buy_price = 0  # Precio de compra inicial
    portfolio_vals = []
    active_positions = []
    short_positions = []
    COM = 0.125/100

    # Comenzar el bucle de backtest
    for index, row in data.iterrows():
        # Señales de indicadores (usar el valor pre-calculado de los indicadores)
        rsi_signal = bollinger_signal = macd_signal_value = cci_signal = stochastic_signal = 0
        
        if "RSI" in strategy:
            if rsi.iloc[index] < rsi_lower:
                rsi_signal = 1  # Señal de compra
            elif rsi.iloc[index] > rsi_upper:
                rsi_signal = -1  # Señal de venta

        if "Bollinger" in strategy:
            if row['Close'] < bollinger.bollinger_lband().iloc[index]:
                bollinger_signal = 1  # Señal de compra
            elif row['Close'] > bollinger.bollinger_hband().iloc[index]:
                bollinger_signal = -1  # Señal de venta
        
        if "MACD" in strategy:
            macd_diff = macd.macd_diff().iloc[index]
            if macd_diff > 0:
                macd_signal_value = 1  # Señal de compra
            elif macd_diff < 0:
                macd_signal_value = -1  # Señal de venta

        if "CCI" in strategy:
            if cci.iloc[index] < -100:
                cci_signal = 1  # Señal de compra
            elif cci.iloc[index] > 100:
                cci_signal = -1  # Señal de venta

        if "Stochastic" in strategy:
            if stochastic.stoch().iloc[index] < 20:  # Línea %K
                stochastic_signal = 1  # Señal de compra
            elif stochastic.stoch().iloc[index] > 80:
                stochastic_signal = -1  # Señal de venta

        # Estrategia de compra/venta simple
        total_signal = rsi_signal + bollinger_signal + macd_signal_value + cci_signal + stochastic_signal

        if total_signal > 0 and len(active_positions) < 100:
            ### There is a buy signal

            # Do we have enough money?
            if capital > row.Close * n_shares * (1 + COM) and len(active_positions) < 100:
                # Buy the share (long position)
                capital -= row.Close * n_shares * (1 + COM)
                active_positions.append(Position(ticker="BTC",
                                                 n_shares=n_shares,
                                                 price=row.Close,
                                                 timestamp=row.Timestamp,
                                                 position_type="long"))

        if total_signal < 0 and len(short_positions) < 100:
            ### There is a short sell signal
            
            max_position_value = capital * 0.1  # Limitar el valor de cada posición al 10% del capital disponible
            if row.Close * n_shares > max_position_value:
                n_shares = max_position_value / row.Close 
            # Do we have enough money?
            if capital > row.Close * n_shares * (1 + COM) and len(short_positions) < 100:
                # Sell short (borrow and sell the share)
                short_positions.append(Position(ticker="BTC",
                                                n_shares=n_shares,
                                                price=row.Close,
                                                timestamp=row.Timestamp,
                                                position_type="short"))

        # Close long positions with SL or TP
        for position in active_positions.copy():
            if row.Close > position.price * (1 + tp):  # Take profit (long)
                capital += row.Close * position.n_shares * (1 - COM)  # Sell and take profit
                active_positions.remove(position)
            elif row.Close < position.price * (1 - sl):  # Stop loss (long)
                capital += row.Close * position.n_shares * (1 - COM)  # Sell at a loss
                active_positions.remove(position)

        # Close short positions with SL or TP
        for position in short_positions.copy():
            if row.Close < position.price * (1 - tp):  # Take profit (short)
                capital += (position.price - row.Close) * position.n_shares * (1 - COM)  # Profit from price decrease
                short_positions.remove(position)
            elif row.Close > position.price * (1 + sl):  # Stop loss (short)
                capital += (position.price - row.Close) * position.n_shares * (1 - COM)  # Loss from price increase
                short_positions.remove(position)

        # Calculate the total value including open positions
        value = capital + sum([position.n_shares * row.Close for position in active_positions]) +                 sum([position.n_shares * (position.price - row.Close) for position in short_positions])

        portfolio_vals.append(value)

    # Close long positions with SL or TP
    for position in active_positions.copy():
        capital += row.Close * position.n_shares * (1 - COM)  # Sell and take profit
        active_positions.remove(position)

    # Close short positions with SL or TP
    for position in short_positions.copy():
        capital += (position.price - row.Close) * position.n_shares * (1 - COM)  # Profit from price decrease
        short_positions.remove(position)


    # Calculate the total value including open positions
    value = capital + sum([position.n_shares * row.Close for position in active_positions]) +             sum([position.n_shares * (position.price - row.Close) for position in short_positions])

    portfolio_vals.append(value)
    
    # Calcular el Sharpe Ratio (anualizado)
    returns = pd.Series(portfolio_vals).pct_change().dropna()

    excess_returns = returns.mean() - risk_free_rate
    sharpe_ratio = (excess_returns / returns.std()) * (252 * 78)**0.5  # Anualización
    print(portfolio_vals[-1])

    return portfolio_vals[-1], sharpe_ratio

# Función de optimización para Optuna
def opt(trial, data, strategy, initial_capital, commission, risk_free_rate):
    # Sugerencias de hiperparámetros mediante Optuna
    stop_loss = trial.suggest_float("stop_loss", 0.005, 0.02)  # Rango más ajustado
    take_profit = trial.suggest_float("take_profit", 0.01, 0.1)
    n_shares = trial.suggest_float("n_shares", 0.1, 17)
    
    # Parámetros para los indicadores técnicos
    rsi_window = rsi_lower = rsi_upper = None
    bollinger_window = bollinger_stddev = None
    macd_fast = macd_slow = macd_signal = None
    cci_window = None
    stochastic_k = stochastic_d = None
    
    if "RSI" in strategy:
        rsi_window = trial.suggest_int("rsi_window", 5, 30)
        rsi_lower = trial.suggest_float("rsi_lower", 20, 50)
        rsi_upper = trial.suggest_float("rsi_upper", 50, 80)
    
    if "Bollinger" in strategy:
        bollinger_window = trial.suggest_int("bollinger_window", 10, 50)
        bollinger_stddev = trial.suggest_float("bollinger_stddev", 1.5, 3.5)
    
    if "MACD" in strategy:
        macd_fast = trial.suggest_int("macd_fast", 12, 26)
        macd_slow = trial.suggest_int("macd_slow", 26, 50)
        macd_signal = trial.suggest_int("macd_signal", 9, 18)
    
    if "CCI" in strategy:
        cci_window = trial.suggest_int("cci_window", 10, 40)
    
    if "Stochastic" in strategy:
        stochastic_k = trial.suggest_int("stochastic_k", 5, 20)
        stochastic_d = trial.suggest_int("stochastic_d", 3, 14)

    # Ejecutar backtest con los parámetros optimizados
    _, sharpe_ratio = backtest(data, strategy, stop_loss, take_profit, n_shares, initial_capital, commission,
                               rsi_window, rsi_lower, rsi_upper,
                               bollinger_window, bollinger_stddev,
                               macd_fast, macd_slow, macd_signal,
                               cci_window, stochastic_k, stochastic_d)

    return sharpe_ratio  # Optuna intentará maximizar el Sharpe Ratio


# Función para guardar los resultados en un archivo CSV
def guardar_resultados_csv(file_path, resultados):
    # Definir las columnas del CSV
    columns = ["strategy", "best_params", "best_sharpe_ratio"]
    
    # Abrir el archivo en modo de escritura
    with open(file_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        
        # Escribir el encabezado
        writer.writeheader()
        
        # Escribir cada fila de resultados
        for result in resultados:
            # Convertir los parámetros del diccionario 'best_params' en un formato de cadena legible
            result["best_params"] = str(result["best_params"])
            writer.writerow(result)
# Función para ejecutar la optimización
def run_optimization(train_data, test_data, combinations, initial_capital, commission, risk_free_rate, n_trials=30):
    best_combination = None
    best_sharpe_ratio = -float('inf')
    results = []
    
    # Para cada combinación de indicadores técnicos
    for strategy in combinations:
        print(f"Optimizando para la combinación: {strategy}")
        
        # Crear un estudio de Optuna para maximizar el Sharpe Ratio
        study = optuna.create_study(direction="maximize")
        
        # Ejecutar la optimización con Optuna en los datos de entrenamiento
        study.optimize(lambda trial: opt(trial, train_data, strategy, initial_capital, commission, risk_free_rate), n_trials=n_trials)
        
        # Guardar los resultados de cada combinación
        results.append({
            "strategy": strategy,
            "best_params": study.best_params,
            "best_sharpe_ratio": study.best_value
        })
        
        # Si esta combinación es la mejor hasta ahora, la guardamos
        if study.best_value > best_sharpe_ratio:
            best_sharpe_ratio = study.best_value
            best_combination = strategy
    
    guardar_resultados_csv("resultados_estrategias.csv", results)

    # Validar la mejor combinación en los datos de prueba
    print("\nValidando la mejor combinación en los datos de prueba...")
    best_params = next(result for result in results if result['strategy'] == best_combination)['best_params']
    
    # Ejecutar backtest con los datos de prueba usando los mejores parámetros
    _, sharpe_ratio_test = backtest(test_data, best_combination, 
                                    best_params['stop_loss'], best_params['take_profit'], 
                                    best_params['n_shares'], initial_capital, commission, 
                                    rsi_window=best_params.get('rsi_window', None),
                                    rsi_lower=best_params.get('rsi_lower', None),
                                    rsi_upper=best_params.get('rsi_upper', None),
                                    bollinger_window=best_params.get('bollinger_window', None),
                                    bollinger_stddev=best_params.get('bollinger_stddev', None),
                                    macd_fast=best_params.get('macd_fast', None),
                                    macd_slow=best_params.get('macd_slow', None),
                                    macd_signal=best_params.get('macd_signal', None),
                                    cci_window=best_params.get('cci_window', None),
                                    stochastic_k=best_params.get('stochastic_k', None),
                                    stochastic_d=best_params.get('stochastic_d', None))

    print(f"Resultados del conjunto de prueba para la mejor combinación ({best_combination}):")
    print(f"Sharpe Ratio en el conjunto de prueba: {sharpe_ratio_test}")

    return best_combination, results, sharpe_ratio_test


# Cargar los datos de entrenamiento y prueba
train_data, test_data = cargar_data("btc_project_train.csv", "btc_project_test.csv")

# Parámetros iniciales
initial_capital = 1_000_000  # Capital inicial
commission = 0.00125  # Comisión del 0.125%
risk_free_rate = 0.05 / (252 * 78)  # Tasa libre de riesgo ajustada a retornos cada 5 minutos

# Indicadores técnicos a combinar
indicators = ['RSI', 'Bollinger', 'MACD', 'CCI', 'Stochastic']

# Generar todas las combinaciones de indicadores técnicos
combinations = generate_combinations(indicators)

# Ejecutar la optimización para todas las combinaciones en el conjunto de entrenamiento y validación en el conjunto de prueba
best_combination, all_results, sharpe_ratio_test = run_optimization(train_data, test_data, combinations, initial_capital, commission, risk_free_rate, n_trials=30)

# Imprimir la mejor combinación y los resultados
print(f"\nLa mejor combinación es: {best_combination}")
for result in all_results:
    print(f"Estrategia: {result['strategy']}, Parámetros: {result['best_params']}, Mejor Sharpe Ratio en entrenamiento: {result['best_sharpe_ratio']}")

print(f"\nSharpe Ratio en el conjunto de prueba: {sharpe_ratio_test}")


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Convert the 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Capital inicial
initial_investment = 1_000_000

# Precio de cierre del primer día
first_close = data.iloc[0]['Close']

# Calcular cuántas acciones se pueden comprar con el capital inicial
shares_bought = initial_investment / first_close
print(f"Acciones compradas el primer día: {shares_bought:.4f}")

# Calcular los retornos diarios
data['Returns'] = data['Close'].pct_change().fillna(0)

# Calcular el valor del portafolio día a día
data['Investment_Value'] = shares_bought * data['Close']

# Graficar el precio de cierre y el valor de la inversión
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Close'], label='Precio de Cierre', color='blue')
plt.title('Precio de Cierre del Activo')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Investment_Value'], label='Valor de la Inversión', color='green')
plt.title('Valor de la Inversión Pasiva (Manteniendo las Acciones)')
plt.xlabel('Fecha')
plt.ylabel('Valor de la Inversión')
plt.legend()
plt.grid(True)
plt.show()

# Valor final de la inversión
final_value = data['Investment_Value'].iloc[-1]
print(f"Valor final de la inversión pasiva: ${final_value:,.2f}")

# Calcular el retorno total de la inversión
passive_return = (final_value - initial_investment) / initial_investment
print(f"Retorno total de la inversión pasiva: {passive_return:.2%}")


# In[ ]:




