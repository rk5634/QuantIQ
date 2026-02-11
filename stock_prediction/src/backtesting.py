
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stock_prediction.config import settings
from stock_prediction.src.utils import setup_logger
from stock_prediction.src.visualization import save_plot

logger = setup_logger("backtesting")

class Backtester:
    def __init__(self, initial_capital=100000, commission=0.001):
        """
        Initialize Backtester.
        
        Args:
            initial_capital: Starting cash.
            commission: Transaction fee per trade (e.g., 0.1%).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
        
    def reset(self):
        self.capital = self.initial_capital
        self.position = 0 # Number of shares
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.equity_curve = []
        self.trades = [] # List of trade dictionaries
        self.entry_date = None
        self.entry_cost = 0.0
        
    def run(self, df: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray = None, model_name: str = "Model"):
        """
        Run backtest simulation.
        
        Args:
            df: DataFrame containing price data.
            predictions: Array of model predictions.
            probabilities: Array of model probabilities (optional, for Sniper Mode).
            model_name: Name of the model.
        """
        self.reset()
        logger.info(f"Starting backtest for {model_name}...")
        
        # Ensure necessary columns exist
        required_cols = ['close', 'high', 'low']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column '{col}' for backtesting.")
                return

        dates = df['time'].values if 'time' in df.columns else df.index
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        # Senior Trader Features (Handle missing if old config used)
        has_senior_features = 'adx' in df.columns and 'atr' in df.columns
        if has_senior_features:
            adx = df['adx'].values
            atr = df['atr'].values
        else:
             logger.warning("ADX/ATR not found. Running in Basic Mode.")
             
        # Loop through data
        # Loop through data
        for i in range(len(df)):
            date = dates[i]
            price = closes[i]
            high = highs[i]
            low = lows[i]
            pred = predictions[i]
            
            # --- Check Exit Conditions (if in position) ---
            if self.position > 0:
                exit_price = None
                reason = ""
                
                # Check Stop Loss (Low hit SL)
                if low <= self.stop_loss:
                    exit_price = self.stop_loss
                    reason = "Stop Loss"
                
                # Check Take Profit (High hit TP)
                # Note: If both hit in same bar, we pessimistically assume SL hit first unless we have minute data
                # Here we check SL first (conservative)
                elif high >= self.take_profit:
                    exit_price = self.take_profit
                    reason = "Take Profit"
                
                if exit_price:
                    # Execute Sell
                    revenue = self.position * exit_price
                    cost = revenue * self.commission
                    
                    # Using current capital logic (Cash + Revenue - Cost)
                    # wait, capital tracks CASH. 
                    self.capital += (revenue - cost)
                    
                    pnl = (revenue - cost) - (self.position * self.entry_price + self.entry_cost)
                    
                    self.trades.append({
                        'entry_date': self.entry_date,
                        'entry_price': self.entry_price,
                        'exit_date': date,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'reason': reason
                    })
                    
                    self.position = 0
                    self.entry_price = 0.0
                    # Position closed, we do not re-enter in the same bar
            
            # --- Check Entry Conditions (if no position) ---
            if self.position == 0 and pred == settings.ACTION_MAPPING['BUY']:
            
                # --- PHASE 10: SNIPER STRATEGY LOGIC ---
                is_sniper_valid = True
                
                # 1. Confidence Check
                if probabilities is not None:
                    # predictions array might match probabilities length
                    if probabilities[i] < settings.SNIPER_CONFIDENCE_THRESHOLD:
                        is_sniper_valid = False
                        
                # 2. Volume Confirmation
                if 'vol_ma_20' in df.columns:
                     # Access volume via values array for speed? 
                     # Should cache values outside loop conceptually, but direct access OK for now
                     # Need to ensure i matches df index? Yes, straight loop.
                     current_vol = df['volume'].values[i]
                     avg_vol = df['vol_ma_20'].values[i]
                     
                     # Check if Valid Volume
                     if not np.isnan(avg_vol) and current_vol < (avg_vol * settings.VOLUME_CONFIRMATION_THRESHOLD):
                          is_sniper_valid = False
                          
                if not is_sniper_valid:
                     continue
                # ---------------------------------------
                
                # Senior Trader Logic: Regime Filter
            
                # Senior Trader Logic: Regime Filter
                if has_senior_features:
                    # If Trend is too weak (Choppy Market), DO NOT BUY
                    if adx[i] < settings.ADX_THRESHOLD:
                         continue 
                
                # Buy Logic
                # Allocate 95% of available capital
                allocation = self.capital * 0.95
                shares = int(allocation / price)
                
                if shares > 0:
                    cost = shares * price
                    comm = cost * self.commission
                    
                    self.capital -= (cost + comm)
                    self.position = shares
                    self.entry_price = price
                    self.entry_cost = comm
                    self.entry_date = date
                    
                    # Set Risk Management
                    # Senior Trader Logic: Dynamic ATR Stop Loss
                    if has_senior_features:
                         self.stop_loss = price - (atr[i] * settings.ATR_MULTIPLIER)
                         # Dynamic TP: Risk:Reward 1:2 usually, or use fixed
                         # Let's keep fixed TP multiplier or use ATR (e.g., 2*ATR risk -> 4*ATR profit?)
                         # User asked for "senior trader" -> typically 1.5x or 2x ATR for TP too, or fixed pct
                         # Let's stick to settings pct for TP for now to isolate SL improvement, or mix.
                         # Better: TP = Price + (atr[i] * settings.ATR_MULTIPLIER * 2) ?
                         # Let's use the explicit settings.TAKE_PROFIT_PCT for consistency unless settings changed.
                         self.take_profit = price * (1 + settings.TAKE_PROFIT_PCT)
                    else:
                        self.stop_loss = price * (1 - settings.STOP_LOSS_PCT)
                        self.take_profit = price * (1 + settings.TAKE_PROFIT_PCT)
            
            # --- Record Equity ---
            # Equity = Cash + Current Market Value of Holdings
            market_value = self.position * price
            total_equity = self.capital + market_value
            self.equity_curve.append({'time': date, 'equity': total_equity})

        # Close open position at end of simulation
        if self.position > 0:
            exit_price = closes[-1]
            revenue = self.position * exit_price
            cost = revenue * self.commission
            self.capital += (revenue - cost)
            
            pnl = (revenue - cost) - (self.position * self.entry_price + self.entry_cost)
            
            self.trades.append({
                'entry_date': self.entry_date,
                'entry_price': self.entry_price,
                'exit_date': dates[-1],
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': "End of Data"
            })
            self.position = 0
            
        # Logging and Visualization
        metrics = self.log_metrics(model_name)
        self.plot_equity_curve(model_name)
        return metrics
        
    def log_metrics(self, model_name):
        df_trades = pd.DataFrame(self.trades)
        metrics = {
            "Model": model_name,
            "Return": "0.00%",
            "Win Rate": "0.00%",
            "Trades": 0,
            "Max Drawdown": "0.00%"
        }

        if df_trades.empty:
            logger.info(f"{model_name} Backtest: No trades executed.")
            return metrics

        total_trades = len(df_trades)
        win_trades = len(df_trades[df_trades['pnl'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        total_pnl = df_trades['pnl'].sum()
        final_capital = self.initial_capital + total_pnl
        return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        df_equity = pd.DataFrame(self.equity_curve)
        if not df_equity.empty:
            df_equity['peak'] = df_equity['equity'].cummax()
            df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
            max_drawdown = df_equity['drawdown'].min() * 100
            
            start_date = df_equity['time'].iloc[0]
            end_date = df_equity['time'].iloc[-1]
            duration_days = (end_date - start_date).days
            duration_str = f"{duration_days} days"
        else:
            max_drawdown = 0.0
            duration_str = "0 days"

        logger.info(f"====== {model_name} BACKTEST RESULTS ======")
        logger.info(f"Duration:        {duration_str}")
        logger.info(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        logger.info(f"Final Capital:   ₹{final_capital:,.2f}")
        logger.info(f"Total Return:    {return_pct:.2f}%")
        logger.info(f"Total Trades:    {total_trades}")
        logger.info(f"Win Rate:        {win_rate:.2%}")
        logger.info(f"Max Drawdown:    {max_drawdown:.2f}%")
        logger.info("==========================================")
        
        metrics.update({
            "Return": f"{return_pct:+.2f}%",
            "Win Rate": f"{win_rate:.2%}",
            "Trades": total_trades,
            "Max Drawdown": f"{max_drawdown:.2f}%",
            "Duration": duration_str
        })
        return metrics
        
    def plot_equity_curve(self, model_name):
        if not self.equity_curve:
            return
        
        df_equity = pd.DataFrame(self.equity_curve)
        plt.figure(figsize=(12, 6))
        plt.plot(df_equity['time'], df_equity['equity'], label='Equity', color='green')
        plt.title(f"Equity Curve - {model_name}")
        plt.xlabel("Date")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(True)
        save_plot(plt.gcf(), f"equity_curve_{model_name.lower().replace(' ', '_')}.png")
