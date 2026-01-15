"""
Market Regime Detection using Hidden Markov Models (HMM)

This module identifies market conditions using a Gaussian HMM:
- Bull: High mean return, low volatility (uptrend)
- Bear: Low/negative mean return, higher volatility (downtrend)
- Chop: Low return, very high volatility (choppy/ranging market)

Based on research:
- Hamilton (1989): Regime-Switching Models for Economic Time Series
- Guidolin & Timmermann (2007): Asset Allocation and Asset Location
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Detect market regimes (Bull, Bear, Chop) using Hidden Markov Models.

    The HMM learns 3 hidden states from returns and volatility:
    1. Bull: High mean return, low variance
    2. Bear: Low/negative mean return, high variance
    3. Chop: Low return, very high variance (indecision)
    """

    def __init__(self, n_components: int = 3, lookback: int = 252):
        """
        Initialize the regime detector.

        Args:
            n_components (int): Number of hidden states (default: 3 = Bull/Bear/Chop)
            lookback (int): Number of periods for training (default: 252 = 1 year)
        """
        self.n_components = n_components
        self.lookback = lookback
        self.hmm = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.state_labels = {}  # Maps state number to label
        self.regime_history = []  # Track regime over time

    def prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare features for HMM: returns and rolling volatility.

        Args:
            df (pd.DataFrame): DataFrame with 'close' column

        Returns:
            np.ndarray: Shape (n_samples, 2) with [returns, volatility]
        """
        if len(df) < self.lookback:
            logger.warning(
                f"Insufficient data: {len(df)} < {self.lookback} lookback period"
            )
            return None

        # Calculate log returns
        df = df.copy()
        df["returns"] = np.log(df["close"] / df["close"].shift(1))

        # Calculate rolling volatility (20-period rolling std)
        df["volatility"] = df["returns"].rolling(window=20).std()

        # Remove NaN values
        df = df.dropna()

        # Create feature matrix [returns, volatility]
        features = df[["returns", "volatility"]].values

        # Standardize features for HMM training
        features_scaled = self.scaler.fit_transform(features)

        return features_scaled

    def train(self, df: pd.DataFrame) -> bool:
        """
        Train the HMM on historical data.

        Args:
            df (pd.DataFrame): Historical OHLCV data with 'close' column

        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            features = self.prepare_features(df)

            if features is None or len(features) < self.lookback:
                logger.error("Failed to prepare features for training")
                return False

            # Train HMM on last `lookback` samples
            features_train = features[-self.lookback :]
            self.hmm.fit(features_train)

            # Decode the hidden states
            hidden_states = self.hmm.predict(features_train)

            # Label states based on mean returns
            self._label_states(features_train, hidden_states)

            self.is_trained = True
            logger.info(f"✓ HMM trained successfully. States: {self.state_labels}")

            return True

        except Exception as e:
            logger.error(f"Failed to train HMM: {e}")
            return False

    def _label_states(self, features: np.ndarray, hidden_states: np.ndarray) -> None:
        """
        Label HMM states as Bull, Bear, or Chop based on mean returns and volatility.

        Args:
            features (np.ndarray): Feature matrix [returns, volatility]
            hidden_states (np.ndarray): Decoded hidden states
        """
        state_means = []

        for state in range(self.n_components):
            mask = hidden_states == state
            if mask.sum() > 0:
                mean_return = features[mask, 0].mean()
                mean_volatility = features[mask, 1].mean()
                state_means.append(
                    {"state": state, "return": mean_return, "vol": mean_volatility}
                )

        # Sort by return (descending)
        state_means.sort(key=lambda x: x["return"], reverse=True)

        # Assign labels
        self.state_labels = {}
        self.state_labels[state_means[0]["state"]] = "bull"  # Highest return
        self.state_labels[state_means[2]["state"]] = "chop"  # Lowest return + high vol
        self.state_labels[state_means[1]["state"]] = "bear"  # Middle return

        logger.debug(f"State labels: {self.state_labels}")

    def predict_regime(self, df: pd.DataFrame) -> str:
        """
        Predict the current market regime.

        Args:
            df (pd.DataFrame): Recent OHLCV data

        Returns:
            str: Current regime ('bull', 'bear', or 'chop')
        """
        if not self.is_trained:
            logger.warning("HMM not trained. Defaulting to 'neutral' regime.")
            return "neutral"

        try:
            features = self.prepare_features(df)

            if features is None or len(features) == 0:
                return "neutral"

            # Predict on the latest 20 samples
            recent_features = features[-20:]
            recent_states = self.hmm.predict(recent_features)

            # Get the most recent state
            current_state = recent_states[-1]

            # Majority vote over last 20 periods for robustness
            most_common_state = np.bincount(recent_states).argmax()

            regime = self.state_labels.get(most_common_state, "neutral")

            # Track history
            self.regime_history.append(
                {"timestamp": df.index[-1], "regime": regime, "confidence": None}
            )

            return regime

        except Exception as e:
            logger.error(f"Failed to predict regime: {e}")
            return "neutral"

    def get_regime_probabilities(self, df: pd.DataFrame) -> dict:
        """
        Get probability distribution over regimes.

        Args:
            df (pd.DataFrame): Recent OHLCV data

        Returns:
            dict: Probabilities for each regime, e.g., {'bull': 0.75, 'bear': 0.20, 'chop': 0.05}
        """
        if not self.is_trained:
            return {"bull": 0.0, "bear": 0.0, "chop": 0.0, "neutral": 1.0}

        try:
            features = self.prepare_features(df)

            if features is None or len(features) == 0:
                return {"bull": 0.0, "bear": 0.0, "chop": 0.0, "neutral": 1.0}

            # Get the latest sample
            latest_feature = features[-1:, :]

            # Get state probabilities
            state_probs = self.hmm.predict_proba(latest_feature)[0]

            # Map to regime names
            probs = {"bull": 0.0, "bear": 0.0, "chop": 0.0}
            for state, prob in enumerate(state_probs):
                regime = self.state_labels.get(state, "neutral")
                if regime != "neutral":
                    probs[regime] = float(prob)

            return probs

        except Exception as e:
            logger.error(f"Failed to get regime probabilities: {e}")
            return {"bull": 0.0, "bear": 0.0, "chop": 0.0, "neutral": 1.0}

    def should_trade(self, regime: str, signal: str) -> bool:
        """
        Determine if a trade should be executed based on regime.

        Logic:
        - Chop: Never trade (kill zone)
        - Bull: Accept BUY, reject SELL
        - Bear: Accept SELL, reject BUY

        Args:
            regime (str): Current regime ('bull', 'bear', 'chop')
            signal (str): Trading signal ('BUY', 'SELL', 'NEUTRAL')

        Returns:
            bool: Whether to execute the trade
        """
        if signal == "NEUTRAL":
            return False

        if regime == "chop":
            return False  # Never trade in chop zone

        if regime == "bull" and signal == "BUY":
            return True  # Buy in bull market

        if regime == "bear" and signal == "SELL":
            return True  # Sell in bear market

        # Reject opposite signals in strong regimes
        return False

    def get_regime_description(self, regime: str) -> str:
        """
        Get human-readable description of regime.

        Args:
            regime (str): Regime label

        Returns:
            str: Descriptive text
        """
        descriptions = {
            "bull": "🟢 BULLISH TREND - Long trades enabled. Market showing uptrend with rising momentum.",
            "bear": "🔴 BEARISH TREND - Short trades enabled. Market showing downtrend with selling pressure.",
            "chop": "⛔ CHOPPY/VOLATILE - Trading paused. Market in indecision zone. Wait for clarity.",
            "neutral": "⚪ NEUTRAL - Insufficient data or training needed.",
        }
        return descriptions.get(regime, "Unknown regime")

    def get_regime_color(self, regime: str) -> str:
        """
        Get color for regime visualization.

        Args:
            regime (str): Regime label

        Returns:
            str: Hex color code
        """
        colors = {
            "bull": "#31a24c",  # Green
            "bear": "#ff6b6b",  # Red
            "chop": "#9b9b9b",  # Grey
            "neutral": "#8b949e",  # Light grey
        }
        return colors.get(regime, "#8b949e")

    def get_status_icon(self, regime: str) -> str:
        """
        Get emoji icon for regime.

        Args:
            regime (str): Regime label

        Returns:
            str: Emoji icon
        """
        icons = {
            "bull": "🐂",
            "bear": "🐻",
            "chop": "🦀",
            "neutral": "⚪",
        }
        return icons.get(regime, "❓")
