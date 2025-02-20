import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame, Series
from typing import Optional, Union
import logging
import tensorflow as tf
import joblib
from joblib import load as joblib_load

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
    
    # Sınıf değişkenleri
    WINDOW_SIZE = 80
    trend_periods = 14
    min_trend_strength = 0.0002
    
    # Özellik listesi
    feature_names = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macdsignal', 'macdhist',
        'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
        'stoch_rsi', 'stoch_k', 'williams_r', 'mom',
        'adx', 'trend_strength'
    ]
    
    # Temel parametreler
    timeframe = '5m'
    minimal_roi = {
        "0": 0.03,
        "30": 0.02,
        "60": 0.01
    }
    
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # Can this strategy go short?
    can_short: bool = False

    # Process only new candles
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

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

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        try:
            # Eager execution'ı etkinleştir
            tf.compat.v1.enable_eager_execution()
            
            # Model yükleme
            with tf.device('/CPU:0'):  # CPU'da çalıştır
                # LSTM katmanı için özel yapılandırma
                custom_objects = {
                    'LSTM': lambda **kwargs: tf.keras.layers.LSTM(
                        **{k: v for k, v in kwargs.items() if k != 'time_major'}
                    )
                }
                self.model = tf.keras.models.load_model(
                    'user_data/models/lstm_model.h5',
                    custom_objects=custom_objects
                )
                
                # Modeli derle
                self.model.compile(optimizer='adam', loss='mse')
                
            logger.info("✓ Model başarıyla yüklendi")
        except Exception as e:
            logger.error(f"❌ Model yüklenirken hata: {e}")
            self.model = None

    def feature_engineering(self, dataframe: DataFrame) -> DataFrame:
        """Tüm özellikleri hazırla"""
        try:
            # RSI
            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            
            # MACD
            macd = ta.MACD(dataframe)
            dataframe['macd'] = macd['macd']
            dataframe['macdsignal'] = macd['macdsignal']
            dataframe['macdhist'] = macd['macdhist']
            
            # Bollinger Bands
            bollinger = ta.BBANDS(dataframe, timeperiod=20)
            dataframe['bb_upper'] = bollinger['upperband']
            dataframe['bb_lower'] = bollinger['lowerband']
            dataframe['bb_middle'] = bollinger['middleband']
            dataframe['bb_width'] = (bollinger['upperband'] - bollinger['lowerband']) / bollinger['middleband']
            
            # Stochastic RSI
            stoch = ta.STOCHRSI(dataframe)
            dataframe['stoch_rsi'] = stoch['fastd']
            dataframe['stoch_k'] = stoch['fastk']
            
            # Williams %R
            dataframe['williams_r'] = ta.WILLR(dataframe)
            
            # Momentum
            dataframe['mom'] = ta.MOM(dataframe['close'], timeperiod=10)
            
            # ADX
            dataframe['adx'] = ta.ADX(dataframe)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"Özellik mühendisliği hatası: {e}")
            return dataframe

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
        """Teknik göstergeleri hesapla"""
        try:
            # Önce temel özellikleri hesapla
            dataframe = self.feature_engineering(dataframe)
            
            # Bollinger Bands
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
            dataframe['bb_upperband'] = bollinger['upper']
            dataframe['bb_middleband'] = bollinger['mid']
            dataframe['bb_lowerband'] = bollinger['lower']
            dataframe['bb_width'] = (
                (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / 
                dataframe['bb_middleband']
            )
            
            # Volatilite hesaplama (20 periyotluk)
            dataframe['volatility'] = dataframe['close'].pct_change().rolling(window=20).std()
            
            # Trend gücü hesaplama
            dataframe['trend_strength'] = self.calculate_trend_strength(dataframe)
            
            # Normalize et
            for feature in self.feature_names:
                if feature in dataframe.columns:
                    mean = dataframe[feature].rolling(window=20).mean()
                    std = dataframe[feature].rolling(window=20).std().replace(0, 1)
                    dataframe[f'{feature}_norm'] = (dataframe[feature] - mean) / std
            
            return dataframe
            
        except Exception as e:
            logger.error(f"Gösterge hesaplama hatası: {e}")
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

    def prepare_data(self, df: DataFrame) -> np.ndarray:
        """LSTM için veriyi hazırla"""
        try:
            # Son WINDOW_SIZE kadar veriyi al
            df_window = df.tail(self.WINDOW_SIZE).copy()
            
            if len(df_window) < self.WINDOW_SIZE:
                logger.warning(f"Yetersiz veri: {len(df_window)} < {self.WINDOW_SIZE}")
                return None
            
            # Normalize et
            df_norm = self.normalize_data(df_window)
            
            # Eksik sütunları kontrol et
            missing_features = [f for f in self.feature_names if f not in df_norm.columns]
            if missing_features:
                logger.error(f"Eksik özellikler: {missing_features}")
                logger.info(f"Mevcut sütunlar: {df_norm.columns.tolist()}")
                return None
            
            # Veriyi numpy dizisine çevir ve yeniden boyutlandır
            data = df_norm[self.feature_names].values
            data = data.reshape(1, self.WINDOW_SIZE, len(self.feature_names))
            
            return data
            
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {e}")
            return None

    def predict_trend(self, dataframe: DataFrame) -> float:
        """LSTM modeli ile trend tahmini yap"""
        if self.model is None:
            return 0
        
        try:
            sequence = self.prepare_data(dataframe)
            if sequence is None:
                return 0
            
            # Birkaç adım ilerisi için tahmin yap
            predictions = []
            current_sequence = sequence.copy()
            
            # 3 adım ilerisi için tahmin yap (15 dakika)
            for _ in range(3):
                pred = self.model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(pred)
                
                # Bir sonraki tahmin için sequence'i güncelle
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred
            
            # Son tahmini al
            final_prediction = predictions[-1]
            
            # Normalize edilmiş tahmin değerini gerçek fiyata dönüştür
            current_close = dataframe['close'].iloc[-1]
            
            # Daha makul bir tahmin için
            price_change_raw = final_prediction * 0.01  # Tahmin değerini yüzdelik değişime çevir
            
            # Momentum ve volatilite bazlı düzeltme
            momentum = dataframe['close'].pct_change(5).iloc[-1]  # 5 periyotluk momentum
            volatility = dataframe['close'].pct_change().std()  # Volatilite
            
            # Düzeltme faktörlerini sınırla
            momentum_factor = max(min(momentum, 0.01), -0.01)  # ±%1 ile sınırla
            volatility_factor = min(volatility, 0.005)  # %0.5 ile sınırla
            
            # Final tahmin
            adjusted_change = price_change_raw + momentum_factor + volatility_factor
            denormalized_prediction = current_close * (1 + adjusted_change)
            
            # Debug için tahmin değerlerini logla
            price_change = ((denormalized_prediction/current_close)-1)*100
            logger.info(f"Tahmin: {denormalized_prediction:.4f}, Mevcut Fiyat: {current_close:.4f}, " 
                       f"Fark: {price_change:.2f}% (Mom: {momentum_factor:.4f}, Vol: {volatility_factor:.4f})")
            
            return float(denormalized_prediction)
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {e}")
            return 0

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Alış sinyallerini oluştur"""
        dataframe.loc[:, 'enter_long'] = 0

        if len(dataframe) >= self.WINDOW_SIZE:
            try:
                prediction = self.predict_trend(dataframe)
                current_close = dataframe['close'].iloc[-1]
                price_change = ((prediction/current_close)-1)*100
                
                # Basit koşullar
                dataframe.loc[
                    (
                        (price_change > 0.1) &  # Minimum %0.1 yükseliş beklentisi
                        (dataframe['rsi'] < self.buy_rsi.value) &  # RSI düşük
                        (dataframe['volume'] > 0) &  # Volume kontrolü
                        (dataframe['macd'] > dataframe['macdsignal'])  # MACD sinyali
                    ),
                    'enter_long'
                ] = 1

                # Log only when signal is generated
                if dataframe['enter_long'].iloc[-1]:
                    logger.info(f"✓ Alış sinyali: {metadata['pair']} - "
                              f"Beklenen: {price_change:.2f}% "
                              f"RSI: {dataframe['rsi'].iloc[-1]:.1f}")
                
            except Exception as e:
                logger.error(f"Sinyal hatası: {e}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Satış sinyallerini oluştur"""
        dataframe.loc[:, 'exit_long'] = 0
        
        if len(dataframe) >= self.WINDOW_SIZE:
            try:
                prediction = self.predict_trend(dataframe)
                current_close = dataframe['close'].iloc[-1]
                
                # Fiyat değişim yüzdesi
                price_change = ((prediction/current_close)-1)*100
                
                # RSI ve diğer indikatörler
                rsi = dataframe['rsi'].iloc[-1]
                momentum = dataframe['close'].pct_change(5).iloc[-1] * 100
                
                # Satış sinyalleri için koşullar
                price_condition = (
                    (price_change < -0.5) |  # %0.5 düşüş beklentisi
                    (price_change > 3.0)     # %3'ten fazla artış beklentisi (kar realizasyonu)
                )
                
                rsi_condition = (
                    (rsi > 80) |  # Aşırı alım
                    (rsi < 20)    # Aşırı satım
                )
                
                trend_condition = (
                    (dataframe['trend_strength'].iloc[-1] < -0.005) &  # Güçlü negatif trend
                    (momentum < -0.2)  # Negatif momentum
                )
                
                # MACD koşulu
                macd_condition = (
                    (dataframe['macd'].iloc[-1] < dataframe['macdsignal'].iloc[-1]) &  # MACD sinyali altında
                    (dataframe['macdhist'].iloc[-1] < 0)  # Negatif histogram
                )
                
                if (price_condition or 
                    (rsi_condition and trend_condition) or 
                    (macd_condition and price_change < 0)):
                    
                    dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                    
                    logger.info(f"✗ Satış sinyali üretildi: {metadata['pair']} - "
                              f"Beklenen Değişim: {price_change:.2f}% "
                              f"RSI: {rsi:.1f} "
                              f"Momentum: {momentum:.2f}%")
                
            except Exception as e:
                logger.error(f"Sinyal üretme hatası: {e}")
        
        return dataframe

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Dinamik stop-loss"""
        # Kar artıkça stop-loss seviyesini yukarı çek
        if current_profit > 0.04:  # %4 kar
            return current_profit - 0.02  # %2 altında stop
        elif current_profit > 0.03:  # %3 kar
            return current_profit - 0.015  # %1.5 altında stop
        elif current_profit > 0.02:  # %2 kar
            return current_profit - 0.01   # %1 altında stop
        
        return self.stoploss  # Varsayılan stop-loss

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                          entry_tag: Optional[str], **kwargs) -> float:
        """
        Özel giriş fiyatı belirleme
        """
        # Volatiliteye göre giriş fiyatını ayarla
        dataframe = self.dp.get_pair_dataframe(pair, self.timeframe)
        volatility = self.get_volatility(dataframe)
        
        if volatility > 0.02:  # Yüksek volatilite
            return proposed_rate * 0.995  # %0.5 daha düşük giriş
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
        """
        Check if entry timeout is reached and cancel order if necessary.
        """
        if order is None:
            return False

        # For "limit" orders, we add a timeout
        # "market" orders should be filled right away
        if order.order_type == 'limit':
            timeout = timedelta(minutes=10)
            if current_time - order.order_date_utc > timeout:
                return True

        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                          current_time: datetime, current_rate: float = 0.0,
                          exit_tag: Optional[str] = None, **kwargs) -> bool:
        """
        Check if exit timeout is reached and cancel order if necessary.
        """
        if order is None:
            return False

        # For "limit" orders, we add a timeout
        # "market" orders should be filled right away
        if order.order_type == 'limit':
            timeout = timedelta(minutes=10)
            if current_time - order.order_date_utc > timeout:
                return True

        return False 
