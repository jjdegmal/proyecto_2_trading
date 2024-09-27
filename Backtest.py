from Strategy import Strategy

def main():
    # Paso 1: Entrenar y optimizar en los datos de entrenamiento
    trading_system = Strategy("5m_train")
    trading_system.run_combinations()
    print("Iniciando optimización de parámetros...")
    trading_system.optimize_parameters()

    # Evaluar la mejor combinación en el dataset de entrenamiento
    trading_system.run_best_combination()

    # Paso 2: Probar la mejor estrategia en los datos de prueba
    trading_system_test = Strategy("5m_test")
    trading_system_test.optimal_combination = trading_system.optimal_combination
    trading_system_test.run_best_combination()

if __name__ == "__main__":
    main()