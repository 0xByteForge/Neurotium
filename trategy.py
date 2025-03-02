import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame, Series
from typing import Optional, Union, Dict, Any
import logging
import tensorflow as tf
import joblib
from joblib import load as joblib_load
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from functools import reduce
import pandas_ta as pta
import os
from tensorflow.keras.regularizers import l2
from pathlib import Path
import talib as ta

logger = logging.getLogger(__name__)

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    DecimalParameter,
    IntParameter,
    RealParameter,
)

import talib.abstract as ta
from technical import qtpylib
from tensorflow.keras.models import load_model

class LstmStrategy1(IStrategy):
    INTERFACE_VERSION = 3
    
    # Temel strateji ayarları

    # ROI için optimize edilebilir parametreler
    roi_t1 = IntParameter(30, 90, default=60, space='roi', optimize=True)
    roi_t2 = IntParameter(91, 180, default=120, space='roi', optimize=True)
    roi_p1 = DecimalParameter(0.01, 0.03, default=0.012, space='roi', optimize=True)
    roi_p2 = DecimalParameter(0.005, 0.015, default=0.008, space='roi', optimize=True)
    roi_p3 = DecimalParameter(0.003, 0.01, default=0.006, space='roi', optimize=True)

    # Başlangıç ROI değerleri (property tarafından override edilecek)
    minimal_roi = {
        "0": 0.012,
        "60": 0.008,
        "120": 0.006
    }

    # Stoploss ve trailing stop ayarları
    stoploss = -0.008
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Futures için gerekli ayarlar
    can_short = False
    position_adjustment_enable = False
    leverage_value = 3  # 3'ten 2'ye düşürüldü
    
    # Startup ve diğer ayarlar
    startup_candle_count = 100
    process_only_new_candles = True

    # Sınıf değişkenleri
    WINDOW_SIZE = 80
    trend_periods = 14
    min_trend_strength = 0.0005
    
    # Özellik listesi
    feature_names = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macdsignal', 'macdhist',
        'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
        'stoch_rsi', 'stoch_k', 'williams_r', 'mom',
        'adx', 'trend_strength', 'price_change'
    ]
    
    # Temel parametreler
    timeframe = '5m'

    # Basit order tipleri
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    # Hyperopt parametreleri
    buy_rsi = IntParameter(low=10, high=40, default=30, space="buy", optimize=True)
    sell_rsi = IntParameter(low=60, high=90, default=70, space="sell", optimize=True)

    # Hyperopt için prediction parametreleri
    buy_prediction_major = DecimalParameter(0.01, 0.10, default=0.05, space='buy', optimize=True)
    buy_prediction_mid = DecimalParameter(0.02, 0.12, default=0.06, space='buy', optimize=True)
    buy_prediction_alt = DecimalParameter(0.03, 0.15, default=0.07, space='buy', optimize=True)
    buy_prediction_meme = DecimalParameter(0.04, 0.18, default=0.08, space='buy', optimize=True)

    # RSI parametreleri - Test sonuçlarına göre optimize edildi
    max_rsi_major = IntParameter(20, 65, default=45, space='buy', optimize=True)  # 20-45 arası en iyi
    max_rsi_mid = IntParameter(35, 50, default=45, space='buy', optimize=True)    # 45 özellikle iyi
    max_rsi_alt = IntParameter(30, 50, default=40, space='buy', optimize=True)    # 33-40-48 seviyeleri
    max_rsi_meme = IntParameter(35, 45, default=40, space='buy', optimize=True)   # 40 seviyesi optimal

    # Exit için RSI değerleri - Tepki seviyeleri
    exit_rsi_major = IntParameter(60, 75, default=65, space='sell', optimize=True)
    exit_rsi_mid = IntParameter(60, 70, default=65, space='sell', optimize=True)
    exit_rsi_alt = IntParameter(70, 80, default=75, space='sell', optimize=True)
    exit_rsi_meme = IntParameter(65, 75, default=70, space='sell', optimize=True)

    # Volume çarpanları
    volume_mult_major = DecimalParameter(0.8, 1.2, default=1.0, space='buy', optimize=True)
    volume_mult_mid = DecimalParameter(0.9, 1.3, default=1.1, space='buy', optimize=True)
    volume_mult_alt = DecimalParameter(1.0, 1.4, default=1.2, space='buy', optimize=True)
    volume_mult_meme = DecimalParameter(1.1, 1.5, default=1.3, space='buy', optimize=True)

    # Exit parametreleri
    exit_prediction_major = DecimalParameter(0.04, 0.08, default=0.06, space='sell', optimize=True)
    exit_prediction_mid = DecimalParameter(0.05, 0.09, default=0.07, space='sell', optimize=True)
    exit_prediction_alt = DecimalParameter(0.06, 0.10, default=0.08, space='sell', optimize=True)
    exit_prediction_meme = DecimalParameter(0.07, 0.11, default=0.09, space='sell', optimize=True)

    # Cooldown parametresi
    cooldown_lookback = IntParameter(6, 24, default=12, space='buy', optimize=True)

    # Stoploss parametreleri
    stoploss_profit_threshold1 = DecimalParameter(-0.03, -0.01, default=-0.02, space='sell', optimize=True)
    stoploss_profit_threshold2 = DecimalParameter(0.01, 0.04, default=0.02, space='sell', optimize=True)
    stoploss_profit_threshold3 = DecimalParameter(0.02, 0.05, default=0.03, space='sell', optimize=True)
    stoploss_margin1 = DecimalParameter(0.001, 0.005, default=0.001, space='sell', optimize=True)
    stoploss_margin2 = DecimalParameter(0.003, 0.007, default=0.005, space='sell', optimize=True)
    stoploss_margin3 = DecimalParameter(0.005, 0.01, default=0.01, space='sell', optimize=True)

    # Timeout parametreleri
    entry_timeout = IntParameter(5, 20, default=10, space='buy', optimize=True)
    exit_timeout = IntParameter(5, 20, default=10, space='sell', optimize=True)

    # EMA parametreleri
    ema_short_period = IntParameter(5, 15, default=8, space='buy', optimize=True)
    ema_long_period = IntParameter(15, 30, default=21, space='buy', optimize=True)

    # Plot config
    plot_config = {
        "main_plot": {
            "tema": {},
            "bb_middleband": {"color": "red"},
            "bb_upperband": {"color": "green"},
            "bb_lowerband": {"color": "green"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            }
        }
    }

    active_trades = {}  # Parite bazlı açık işlemleri takip et

    # Maksimum açık kalma süresi
    max_open_trades_duration = timedelta(hours=24)  # Maksimum 24 saat

    # Coin kategorileri
    MAJOR_PAIRS = ['BTC/USDT', 'ETH/USDT']
    MID_PAIRS = ['BNB/USDT', 'SOL/USDT']
    ALT_PAIRS = ['XRP/USDT', 'ADA/USDT']
    MEME_PAIRS = ['DOGE/USDT', 'SHIB/USDT']
    
    # Her kategori için parametreler
 #   PAIR_PARAMS = {
 #       'MAJOR': {
 #           'min_prediction': 0.08,
 #           'max_rsi': 60,
 #           'volume_mult': 1.3,
 #           'exit_prediction': 0.06,
 #           'exit_rsi': 75
 #       },
 #       'MID': {
 #           'min_prediction': 0.09,
 #           'max_rsi': 55,
 #           'volume_mult': 1.4,
 #           'exit_prediction': 0.07,
 #           'exit_rsi': 70
 #       },
 #       'ALT': {
 #           'min_prediction': 0.10,
 #           'max_rsi': 50,
 #           'volume_mult': 1.5,
 #           'exit_prediction': 0.08,
 #           'exit_rsi': 65
 #       },
 #       'MEME': {
 #           'min_prediction': 0.12,
 #           'max_rsi': 45,
 #           'volume_mult': 2.0,
 #           'exit_prediction': 0.09,
 #           'exit_rsi': 60
 #       }
 #   }

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        try:
            # GPU kullanımını devre dışı bırak
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            
            logger.info(f"TensorFlow sürümü: {tf.__version__}")
            
            # Model yolu düzeltmesi
            current_path = Path(__file__).parent
            model_path = current_path / "models" / "finalmodel.h5"
            
            logger.info(f"Model yolu: {model_path}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
            
            try:
                # Model yükleme
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
                # Model'i yeniden derle
                self.model.compile(optimizer='adam', loss='mse')
                logger.info("✓ Model başarıyla yüklendi")
            except Exception as model_error:
                logger.error(f"Model yükleme hatası detayı: {str(model_error)}")
                raise
            
        except Exception as e:
            logger.error(f"❌ Model yüklenirken hata: {str(e)}")
            logger.info("Basit strateji modunda devam ediliyor...")
            self.model = None

    def calculate_features(self, dataframe: DataFrame) -> DataFrame:
        try:
            # RSI
            dataframe['rsi'] = ta.RSI(dataframe['close'])
            
            # MACD
            macd, macdsignal, macdhist = ta.MACD(dataframe['close'])
            dataframe['macd'] = macd
            dataframe['macdsignal'] = macdsignal
            dataframe['macdhist'] = macdhist
            
            # Bollinger Bands
            upperband, middleband, lowerband = ta.BBANDS(dataframe['close'])
            dataframe['bb_upper'] = upperband
            dataframe['bb_middle'] = middleband
            dataframe['bb_lower'] = lowerband
            dataframe['bb_width'] = (upperband - lowerband) / middleband
            
            # Stochastic RSI
            fastk, fastd = ta.STOCHRSI(dataframe['close'])
            dataframe['stoch_rsi'] = fastd
            dataframe['stoch_k'] = fastk
            
            # Williams %R
            dataframe['williams_r'] = ta.WILLR(dataframe['high'], dataframe['low'], dataframe['close'])
            
            # ADX
            dataframe['adx'] = ta.ADX(dataframe['high'], dataframe['low'], dataframe['close'])
            
            # Trend Strength (20 periyotluk fiyat değişimi)
            dataframe['trend_strength'] = abs(dataframe['close'].pct_change(20))
            
            # Short Pressure
            dataframe['short_pressure'] = ((dataframe['close'] < dataframe['bb_lower']) & 
                                         (dataframe['rsi'] < 30) & 
                                         (dataframe['williams_r'] < -80)).astype(int)
            
            # Trend Reversal
            dataframe['trend_reversal'] = ((dataframe['macd'] < dataframe['macdsignal']) & 
                                         (dataframe['stoch_rsi'] > 0.8)).astype(int)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"Feature engineering hatası: {str(e)}")
            raise

    def calculate_trend_strength(self, dataframe: DataFrame) -> Series:
        """Trend gücünü hesapla"""
        try:
            # Trend gücü = (Mevcut fiyat - n periyot önceki fiyat) / n periyot önceki fiyat
            return (
                dataframe['close'] - dataframe['close'].shift(self.trend_periods)
            ) / dataframe['close'].shift(self.trend_periods)
        except Exception as e:
            logger.error(f"Trend gücü hesaplama hatası: {e}")
            return pd.Series(0, index=dataframe.index)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Temel göstergeleri hesapla
        for col in self.feature_names:
            if col not in dataframe.columns:
                dataframe[col] = 0
        
        # Sinyal kolonlarını başlat
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        
        return dataframe

    def normalize_data(self, df: DataFrame) -> DataFrame:
        """Verileri normalize et"""
        result = df.copy()
        for column in self.feature_names:
            if column in result.columns:
                mean = result[column].mean()
                std = result[column].std()
                if std == 0:  # Sıfıra bölünmeyi önle
                    std = 1
                result[column] = (result[column] - mean) / std
        return result

    def prepare_data_for_model(self, dataframe: DataFrame) -> np.ndarray:
        try:
            # Önce teknik göstergeleri hesapla
            dataframe = self.calculate_features(dataframe)
            
            # Model için gerekli özellikler
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macdsignal', 'macdhist',
                'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
                'stoch_rsi', 'stoch_k', 'williams_r',
                'adx', 'trend_strength', 'short_pressure', 'trend_reversal'
            ]
            
            # Eksik değerleri doldur
            dataframe = dataframe.ffill()  # DataFrame.fillna yerine ffill() kullan
            dataframe = dataframe.fillna(0)
            
            # Özellikleri normalize et
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(dataframe[features].values)
            
            # Son 80 veriyi al ve yeniden şekillendir
            window_size = 80
            last_window = normalized_data[-window_size:]
            
            # Eğer yeterli veri yoksa, sıfırlarla doldur
            if len(last_window) < window_size:
                padding = np.zeros((window_size - len(last_window), len(features)))
                last_window = np.vstack([padding, last_window])
            
            # Veriyi (1, 80, 20) şekline getir
            model_input = last_window.reshape(1, window_size, len(features))
            
            logger.info(f"Model input boyutu: {model_input.shape}")
            return model_input
            
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            raise

    def predict_trend(self, dataframe: DataFrame) -> float:
        try:
            if self.model is None:
                return 0
            
            # Veriyi hazırla
            prepared_data = self.prepare_data_for_model(dataframe)
            
            # Tahmin yap
            prediction = self.model.predict(prepared_data, verbose=0)
            
            # Tahmin değerini logla
            logger.info(f"Model tahmini: {prediction[0][0]:.4f}")
            
            return prediction[0][0]
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return 0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            prediction = self.predict_trend(dataframe)
            pair = metadata['pair']
            
            # RSI hesapla
            dataframe['rsi'] = ta.RSI(dataframe['close'])
            
            # Pair kategorisine göre parametreleri ayarla
            if any(p in pair for p in self.MAJOR_PAIRS):
                pred_threshold = self.buy_prediction_major.value
                max_rsi = self.max_rsi_major.value
                logger.info(f"MAJOR {pair} - Prediction: {prediction:.4f}, Threshold: {pred_threshold:.4f}")
            elif any(p in pair for p in self.MID_PAIRS):
                pred_threshold = self.buy_prediction_mid.value
                max_rsi = self.max_rsi_mid.value
                logger.info(f"MID {pair} - Prediction: {prediction:.4f}, Threshold: {pred_threshold:.4f}")
            elif any(p in pair for p in self.ALT_PAIRS):
                pred_threshold = self.buy_prediction_alt.value
                max_rsi = self.max_rsi_alt.value
                logger.info(f"ALT {pair} - Prediction: {prediction:.4f}, Threshold: {pred_threshold:.4f}")
            elif any(p in pair for p in self.MEME_PAIRS):
                pred_threshold = self.buy_prediction_meme.value
                max_rsi = self.max_rsi_meme.value
                logger.info(f"MEME {pair} - Prediction: {prediction:.4f}, Threshold: {pred_threshold:.4f}")
            
            # Tüm kategoriler için ortak giriş koşulları
            long_conditions = (
                (prediction > pred_threshold) &  # Prediction kontrolü
                (dataframe['volume'] > 0) &  # Volume kontrolü
                (dataframe['rsi'] < max_rsi) &  # RSI kontrolü
                (dataframe['close'] > dataframe['close'].shift(1))  # Yükselen trend
            )
            
            # Cooldown süresini kısaltalım
            cooldown = (dataframe['volume'] > 0)
            for i in range(2):
                cooldown &= (dataframe['enter_long'].shift(i).fillna(0) == 0)
            
            dataframe.loc[long_conditions & cooldown, 'enter_long'] = 1
            
        except Exception as e:
            logger.error(f"Sinyal üretme hatası: {str(e)}")
            
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        try:
            prediction = self.predict_trend(dataframe)
            pair = metadata['pair']
            
            # Temel göstergeler - EMA periyotları için int dönüşümü yapıyoruz
            dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=int(self.ema_short_period.value))
            dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=int(self.ema_long_period.value))
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            
            # Pair'e göre özel çıkış koşulları
            if 'BTC/USDT' in pair or 'ETH/USDT' in pair:  # Major
                exit_conditions = (
                    (prediction < self.exit_prediction_major.value) |
                    (dataframe['rsi'] > self.exit_rsi_major.value) |
                    (dataframe['ema_short'] < dataframe['ema_long'])
                )
            
            elif 'BNB/USDT' in pair or 'SOL/USDT' in pair:  # Mid
                exit_conditions = (
                    (prediction < self.exit_prediction_mid.value) |
                    (dataframe['rsi'] > self.exit_rsi_mid.value)
                )
            
            elif 'XRP/USDT' in pair or 'ADA/USDT' in pair:  # Alt
                exit_conditions = (
                    (prediction < self.exit_prediction_alt.value) |
                    (dataframe['rsi'] > self.exit_rsi_alt.value)
                )
            
            elif 'DOGE/USDT' in pair or 'SHIB/USDT' in pair:  # Meme
                exit_conditions = (
                    (prediction < self.exit_prediction_meme.value) |
                    (dataframe['rsi'] > self.exit_rsi_meme.value)
                )
            
            dataframe.loc[exit_conditions, 'exit_long'] = 1
            
        except Exception as e:
            logger.error(f"Çıkış sinyali üretme hatası: {str(e)}")
            
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        # Zarar durumunda daha hızlı çık - artık optimize edilebilir
        if current_profit < self.stoploss_profit_threshold1.value:
            return self.stoploss_margin1.value
        
        # Kâr durumunda trailing stop - artık optimize edilebilir
        if current_profit > self.stoploss_profit_threshold3.value:
            return current_profit - self.stoploss_margin3.value
        elif current_profit > self.stoploss_profit_threshold2.value:
            return current_profit - self.stoploss_margin2.value
        
        return self.stoploss

    def custom_entry_price(self, pair: str, current_time: datetime,
                         proposed_rate: float, entry_tag: Optional[str],
                         side: str, **kwargs) -> float:
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: Trade,
                         current_time: datetime, proposed_rate: float,
                         current_profit: float, exit_tag: Optional[str],
                         **kwargs) -> float:
        return proposed_rate

    def get_volatility(self, dataframe: pd.DataFrame) -> float:
        """
        Volatilite hesaplama
        """
        return np.std(dataframe['close'].pct_change().dropna()) * np.sqrt(288)  # Günlük volatilite
        
    def get_trend_strength(self, dataframe: pd.DataFrame) -> float:
        """
        Trend gücünü hesapla
        """
        close_prices = dataframe['close'].values
        trend = (close_prices[-1] - close_prices[-self.trend_periods]) / close_prices[-self.trend_periods]
        return trend

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                           current_time: datetime, current_rate: float = 0.0, 
                           entry_tag: Optional[str] = None, **kwargs) -> bool:
        if order is None:
            return False

        if order.order_type == 'limit':
            timeout = timedelta(minutes=self.entry_timeout.value)  # Artık optimize edilebilir
            if current_time - order.order_date_utc > timeout:
                return True

        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                          current_time: datetime, current_rate: float = 0.0,
                          exit_tag: Optional[str] = None, **kwargs) -> bool:
        if order is None:
            return False

        if order.order_type == 'limit':
            timeout = timedelta(minutes=self.exit_timeout.value)  # Artık optimize edilebilir
            if current_time - order.order_date_utc > timeout:
                return True

        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          side: str, **kwargs) -> bool:
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                         rate: float, time_in_force: str, exit_reason: str,
                         current_time: datetime, **kwargs) -> bool:
        return True

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        return proposed_stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """Sabit kaldıraç değeri"""
        return self.leverage_value  # leverage_value'yu döndür

    def get_features(self, dataframe: DataFrame) -> np.ndarray:
        """Özellik vektörünü hazırla"""
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macdsignal', 'macdhist',
            'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
            'stoch_rsi', 'stoch_k', 'williams_r',
            'adx', 'trend_strength', 'short_pressure', 'trend_reversal'
        ]
        
        return dataframe[features].values 
