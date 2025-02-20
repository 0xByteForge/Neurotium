import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import ta

def normalize_crypto_data(input_file):
    print(f"\n{input_file} işleniyor...")
    
    # CSV dosyasını oku
    df = pd.read_csv(input_file)
    
    # Temel fiyat ve hacim verilerini normalize et
    price_volume_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # MinMaxScaler ile fiyat ve hacim normalize et
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[price_volume_columns] = scaler.fit_transform(df[price_volume_columns])
    
    # Teknik göstergeleri normalize et
    technical_columns = [col for col in df.columns if col not in ['timestamp'] + price_volume_columns]
    
    if technical_columns:
        # Teknik göstergeler için StandardScaler kullan
        tech_scaler = StandardScaler()
        df[technical_columns] = tech_scaler.fit_transform(df[technical_columns].fillna(0))
    
    # NaN değerleri temizle
    df = df.fillna(0)
    
    # Normalize edilmiş veriyi kaydet
    output_file = 'normalized_' + input_file
    df.to_csv(output_file, index=False)
    print(f"Normalize edilmiş veri kaydedildi: {output_file}")
    
    return df

def add_technical_indicators(df):
    # Temel göstergeler
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['rsi_4h'] = ta.momentum.rsi(df['close'], window=48)
    
    # MACD varyasyonları
    macd = ta.trend.macd(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands ve türevleri
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Çoklu periyot hareketli ortalamalar
    for period in [20, 50, 100, 200]:
        df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
    
    # Momentum göstergeleri
    df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    df['ultimate_oscillator'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
    
    # Trend göstergeleri
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
    
    return df

def normalize_data(df):
    # Log-return hesapla
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatilite hesapla
    df['volatility'] = df['return'].rolling(window=20).std()
    
    # Z-score normalizasyonu
    scaler = StandardScaler()
    columns_to_normalize = ['open', 'high', 'low', 'close', 'volume']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    
    return df

def main():
    # Normalize edilecek dosyalar
    market_types = ['spot']  # Sadece spot
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'BNBUSDT']
    timeframe = '5m'
    
    # Spot verileri işle
    spot_files = [f'binance_{symbol}_{timeframe}_data.csv' for symbol in symbols]
    for file in spot_files:
        try:
            df = normalize_crypto_data(file)
            print(f"Spot - İşlenen satır sayısı: {len(df)}")
        except Exception as e:
            print(f"Hata: {file} işlenirken bir sorun oluştu - {str(e)}")

if __name__ == "__main__":
    main() 
