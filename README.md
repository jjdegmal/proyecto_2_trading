# Technical Analysis Project

## Descripción  
This project optimizes trading strategies by applying five technical indicators to historical data in Python. Buy/sell signals are generated, tested with backtesting, and key parameters are optimized using advanced techniques. Results include graphs and metrics compared to a passive strategy.

## Estructura del proyecto 
- **data/**: Contains training and test datasets for different timeframes.
- **technical_analysis/**: Contains only module specific code.
- **utils/**: Helper methods, etc.
- **report.ipynb**: Visualizations, tables & conclusions in Jupyter Notebook.
- **venv/**: Virtual environment.
- **.gitignore**: Python's gitignore file from GitHub.
- **README.md**: containing the description of the project, the steps required to run the main code. 
- **requirements.txt**: Libraries and versions required to run the module.

## Usage

1. Establish an environment in Python.
2. Install the required libraries `requirements.txt`.
3. Run the main code `CODE.py`:
4. Prepare the Datasets
You need to have the following CSV files ready:

- aapl_5m_train.csv (for training)
- aapl_5m_test.csv (for testing)
Make sure these files contain the necessary columns, including Timestamp, Close, High, Low.

5. Load and Run the Code
Load the datasets by specifying the correct file paths to your training and testing data.
Define the initial parameters:
Initial capital (initial_capital), e.g., $1,000,000.
Commission rate (commission), e.g., 0.125% (0.00125).
Risk-free rate (risk_free_rate), calculated based on annual return.
Generate combinations of technical indicators to test various strategies.
Run the optimization using Optuna, which will test different parameter sets for each combination of technical indicators. The goal is to maximize the Sharpe Ratio.
Backtest the best-performing strategy on test data to validate its performance.
Visualize the results by plotting the portfolio value over time for each strategy compared to a passive buy-and-hold approach.

6. Running the Optimization
Run the optimization function, which will:

Iterate through different combinations of indicators.
Use Optuna to optimize the parameters like stop-loss, take-profit, and indicator settings.
Backtest each strategy and select the best-performing one based on the Sharpe Ratio.

7. Interpreting the Results
Once the optimization is complete:

The best strategy combination will be identified.
You will get the optimal parameters (stop-loss, take-profit, etc.).
The results will include the Sharpe Ratio for both training and testing datasets.

8. Plotting Performance
After validation, you can compare the best strategies' performance against a passive buy-and-hold strategy. The resulting plot will display portfolio growth over time for easy comparison.

9. Adjust Parameters and Retry
If necessary, adjust the number of optimization trials or fine-tune your dataset and re-run the optimization.



## Autores
- Gustavo Guevara López
- José Jorge Degollado Maldonado
- Erick Pacheco Parra
