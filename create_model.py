import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import gc
from datetime import datetime
import ta
import psutil
from psutil import Process
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import seaborn as sns

# Gerekli importlar ve tanımlamalar
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Dizin tanımlamaları
models_dir = os.path.join('user_data', 'models')
checkpoint_dir = os.path.join(models_dir, 'checkpoints', datetime.now().strftime("%Y%m%d-%H%M%S"))
final_model_path = os.path.join(models_dir, f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
os.makedirs(checkpoint_dir, exist_ok=True)

# GPU ayarlarını en başa ekle
print("GPU Ayarları Yapılandırılıyor...")
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# GPU'ları listele ve yapılandır
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU devices found")

# Process nesnesini oluştur
process = psutil.Process()

# Dizin yolları - tam yol kullanarak
data_dir = r'c:\Users\casper\Desktop\Proje\freqtrade\ft_userdata\user_data\data\binance\futures'
base_dir = os.path.dirname(os.path.dirname(data_dir))  # data_dir'den iki üst dizine çık
plots_dir = os.path.join(base_dir, 'plots')
log_dir = os.path.join(base_dir, 'logs')
models_dir = os.path.join(base_dir, 'models')

# Dizinleri oluştur
for dir_path in [data_dir, models_dir, plots_dir, log_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Model parametreleri
WINDOW_SIZE = 80
FUTURE_STEPS = 1
STRIDE = 1
BATCH_SIZE = 64
CHUNK_SIZE = 50000  # Küçük chunk boyutu

# İşlenecek kripto para çiftleri
pairs = [
    'BTC_USDT_USDT', 'ETH_USDT_USDT', 'BNB_USDT_USDT',
    'SOL_USDT_USDT', 'XRP_USDT_USDT', 'ADA_USDT_USDT',
    'DOGE_USDT_USDT'
]

# İlk veri yükleme kısmını kaldır
# print("Veri yükleniyor...")
# pairs = [ ... ]
# all_data = []
# for pair in pairs: ...

# Sliding window ve veri hazırlama fonksiyonlarını birleştir
def prepare_data_for_training(df: pd.DataFrame) -> tuple:
    """
    Long ve Short pozisyonlar için veriyi hazırla
    """
    print("Veriyi hazırlama...")
    
    # Temel fiyat ve hacim verileri
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    
    print("Teknik göstergeler hesaplanıyor...")
    # Numpy array'lere dönüştür - daha hızlı işlem için
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # RSI ve MACD
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macdsignal'] = macd.macd_signal()
    df['macdhist'] = macd.macd_diff()
    
    # Bollinger Bands - numpy ile hesapla
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic RSI ve Williams %R
    stoch = ta.momentum.StochRSIIndicator(df['close'])
    df['stoch_rsi'] = stoch.stochrsi_d()
    df['stoch_k'] = stoch.stochrsi_k()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
    
    # ADX ve Trend Strength
    adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx_indicator.adx()
    df['trend_strength'] = df['adx']
    
    # Dinamik eşik hesaplama
    def calculate_futures_dynamic_threshold(df, base_threshold=0.003):
        # Volatilite hesaplama (daha kısa pencere kullanıyoruz)
        volatility = df['close'].pct_change().rolling(window=50).std()
        
        # ATR hesaplama (daha kısa pencere)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.DataFrame({'hl':high_low, 'hc':high_close, 'lc':low_close}).max(axis=1)
        atr = true_range.rolling(window=10).mean()
        
        # Hacim bazlı momentum
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_momentum = df['volume'] / volume_ma
        
        # Fiyat momentum göstergesi
        price_momentum = df['close'].pct_change(5).abs()
        
        # Dinamik eşik hesaplama
        dynamic_threshold = base_threshold * (
            0.3 * (volatility / volatility.rolling(100).mean()) +  # Volatilite etkisi
            0.3 * (atr / df['close']) +                           # ATR etkisi
            0.2 * volume_momentum +                               # Hacim etkisi
            0.2 * (1 + price_momentum)                           # Momentum etkisi
        )
        
        # Eşik sınırlarını belirle (futures için daha geniş aralık)
        return np.clip(dynamic_threshold, base_threshold * 0.4, base_threshold * 4)
    
    # Dinamik eşikleri hesapla
    dynamic_thresholds = calculate_futures_dynamic_threshold(df)
    
    # Fiyat değişimlerini ve hedefleri hesapla
    df['price_change'] = df['close'].pct_change(periods=FUTURE_STEPS)
    
    # Yeni özellikler ekle
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['trend_strength'] = abs(df['close'].pct_change(20))
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Daha sıkı filtreleme
    df['strong_trend'] = (df['trend_strength'] > df['trend_strength'].quantile(0.7)).astype(int)
    df['high_vol'] = (df['volatility'] > df['volatility'].quantile(0.7)).astype(int)
    
    # Hedefleri güncelle
    df['long_target'] = ((df['price_change'] > dynamic_thresholds) & 
                        (df['strong_trend'] == 1)).astype(int)
    df['short_target'] = ((df['price_change'] < -dynamic_thresholds) & 
                         (df['strong_trend'] == 1)).astype(int)
    
    # Short pozisyonlar için daha hassas eşikler
    df['short_target'] = ((df['price_change'] < -dynamic_thresholds * 0.8) &  # Eşik düşürüldü
                         (df['strong_trend'] == 1) &
                         (df['volatility'] > df['volatility'].rolling(50).mean()) &  # Volatilite kontrolü
                         (df['volume'] > df['volume'].rolling(20).mean() * 1.2)  # Hacim kontrolü
                         ).astype(int)
    
    # Short sinyalleri için ek özellikler
    df['short_pressure'] = ((df['close'] < df['bb_lower']) & 
                          (df['rsi'] < 30) & 
                          (df['williams_r'] < -80)).astype(int)
    
    df['trend_reversal'] = ((df['macd'] < df['macdsignal']) & 
                          (df['stoch_rsi'] > 0.8)).astype(int)
    
    features = ['open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'macdsignal', 'macdhist',
                'bb_upper', 'bb_lower', 'bb_middle', 'bb_width',
                'stoch_rsi', 'stoch_k', 'williams_r',
                'adx', 'trend_strength', 'short_pressure', 'trend_reversal']
    
    print("Sliding windows oluşturuluyor...")
    X = df[features].values
    y_long = df['long_target'].values
    y_short = df['short_target'].values
    
    total_windows = len(X) - WINDOW_SIZE
    chunk_size = 100000  # İşleme chunk boyutu
    num_chunks = (total_windows + chunk_size - 1) // chunk_size
    
    print(f"Toplam {num_chunks} chunk işlenecek...")
    print(f"Toplam veri sayısı: {len(X)}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Chunk size: {chunk_size}")
    
    X_windows_list = []
    y_long_list = []
    y_short_list = []
    
    try:
        for chunk in range(num_chunks):
            start_idx = chunk * chunk_size
            end_idx = min((chunk + 1) * chunk_size, total_windows)
            
            print(f"\nChunk {chunk + 1}/{num_chunks} işleniyor...")
            print(f"Start index: {start_idx}")
            print(f"End index: {end_idx}")
            
            # Her chunk için windows oluştur
            chunk_windows = []
            chunk_y_long = []
            chunk_y_short = []
            
            for i in range(start_idx, end_idx):
                if i % 10000 == 0:  # Her 10000 örnekte bir durum raporu
                    print(f"İşlenen örnek: {i - start_idx}/{end_idx - start_idx}")
                
                window = X[i:i + WINDOW_SIZE]
                if len(window) == WINDOW_SIZE:  # Window size kontrolü
                    chunk_windows.append(window)
                    chunk_y_long.append(y_long[i + WINDOW_SIZE])
                    chunk_y_short.append(y_short[i + WINDOW_SIZE])
            
            if chunk_windows:  # Boş chunk kontrolü
                chunk_windows = np.array(chunk_windows, dtype=np.float32)
                chunk_y_long = np.array(chunk_y_long, dtype=np.int8)
                chunk_y_short = np.array(chunk_y_short, dtype=np.int8)
                
                X_windows_list.append(chunk_windows)
                y_long_list.append(chunk_y_long)
                y_short_list.append(chunk_y_short)
        
                print(f"Chunk {chunk + 1} tamamlandı")
                print(f"Chunk boyutu: {len(chunk_windows)}")
                try:
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    print(f"Bellek kullanımı: {memory_usage:.2f} MB")
                except Exception as e:
                    print(f"Bellek kullanımı ölçülemedi: {e}")
            
            # Her chunk sonrası bellek temizliği
            del chunk_windows, chunk_y_long, chunk_y_short
            gc.collect()
        
        print("\nTüm chunk'lar işlendi, veriler birleştiriliyor...")
        
        if not X_windows_list:  # Boş liste kontrolü
            raise ValueError("Hiç veri penceresi oluşturulamadı!")
        
        print("X_windows_list uzunluğu:", len(X_windows_list))
        print("Her bir chunk şekli:")
        for i, chunk in enumerate(X_windows_list):
            print(f"Chunk {i+1}: {chunk.shape}")
        
        # Parça parça birleştirme kısmını güncelle
        print("\nVeriler parça parça birleştiriliyor...")

        try:
            # Geçici dosya dizini oluştur
            temp_dir = os.path.join(base_dir, 'temp_arrays')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Her chunk'ı ayrı dosyaya kaydet
            print("Chunk'lar diske kaydediliyor...")
            total_size = 0
            for i, (X_chunk, y_long_chunk, y_short_chunk) in enumerate(zip(X_windows_list, y_long_list, y_short_list)):
                chunk_size = len(X_chunk)
                total_size += chunk_size
                
                # Dosya yolları
                x_path = os.path.join(temp_dir, f'x_chunk_{i}.npy')
                y_long_path = os.path.join(temp_dir, f'y_long_chunk_{i}.npy')
                y_short_path = os.path.join(temp_dir, f'y_short_chunk_{i}.npy')
                
                # Numpy array'leri kaydet
                np.save(x_path, X_chunk)
                np.save(y_long_path, y_long_chunk)
                np.save(y_short_path, y_short_chunk)
                
                print(f"Chunk {i+1} kaydedildi. Boyut: {chunk_size}")
                
                # Belleği temizle
                X_windows_list[i] = None
                y_long_list[i] = None
                y_short_list[i] = None
                gc.collect()
            
            # Listeleri temizle
            del X_windows_list, y_long_list, y_short_list
            gc.collect()
            
            # Final array'leri oluştur
            print("\nFinal array'ler oluşturuluyor...")
            X_windows = np.zeros((total_size, WINDOW_SIZE, len(features)), dtype=np.float32)
            y_long_windows = np.zeros(total_size, dtype=np.int8)
            y_short_windows = np.zeros(total_size, dtype=np.int8)
            
            # Parça parça yükle ve birleştir
            current_idx = 0
            for i in range(16):  # Toplam chunk sayısı
                print(f"\nChunk {i+1}/16 yükleniyor ve birleştiriliyor...")
                
                # Dosyaları yükle
                x_path = os.path.join(temp_dir, f'x_chunk_{i}.npy')
                y_long_path = os.path.join(temp_dir, f'y_long_chunk_{i}.npy')
                y_short_path = os.path.join(temp_dir, f'y_short_chunk_{i}.npy')
                
                X_chunk = np.load(x_path)
                y_long_chunk = np.load(y_long_path)
                y_short_chunk = np.load(y_short_path)
                
                chunk_size = len(X_chunk)
                end_idx = current_idx + chunk_size
                
                # Array'lere ekle
                X_windows[current_idx:end_idx] = X_chunk
                y_long_windows[current_idx:end_idx] = y_long_chunk
                y_short_windows[current_idx:end_idx] = y_short_chunk
                
                # İndeksi güncelle
                current_idx = end_idx
                
                # Geçici dosyaları sil
                os.remove(x_path)
                os.remove(y_long_path)
                os.remove(y_short_path)
                
            
            # Geçici dizini sil
            os.rmdir(temp_dir)
            
            print(f"\nSon veri şekilleri:")
            print(f"X: {X_windows.shape}")
            print(f"y_long: {y_long_windows.shape}")
            print(f"y_short: {y_short_windows.shape}")
            
            # Veri dengeleme
            print("\nVeri dengeleme başlıyor...")
            X_windows, y_long_windows, y_short_windows = prepare_balanced_data(
                X_windows, y_long_windows, y_short_windows
            )
            
            print(f"\nDengelenmiş veri boyutları:")
            print(f"X: {X_windows.shape}")
            print(f"y_long: {y_long_windows.shape}")
            print(f"y_short: {y_short_windows.shape}")
            
            # Veri normalizasyonu ekle
            def normalize_features(X):
                """
                Güvenli normalizasyon - NaN kontrolü ile
                """
                # NaN kontrolü
                if np.isnan(X).any():
                    print("Uyarı: Veride NaN değerler bulundu. Temizleniyor...")
                    X = np.nan_to_num(X, nan=0.0)
                
                # Robust normalizasyon
                for i in range(X.shape[2]):  # Her özellik için
                    feature = X[:, :, i]
                    q1 = np.percentile(feature, 25)
                    q3 = np.percentile(feature, 75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    
                    # Aykırı değerleri sınırla
                    feature = np.clip(feature, lower, upper)
                    
                    # Min-max normalizasyon
                    feature_min = np.min(feature)
                    feature_max = np.max(feature)
                    feature_range = feature_max - feature_min
                    if feature_range == 0:
                        feature_range = 1
                    
                    X[:, :, i] = (feature - feature_min) / feature_range
                
                return X
            
            X_windows = normalize_features(X_windows)
            
            return X_windows, y_long_windows, y_short_windows, features

        except Exception as e:
            print(f"\nBirleştirme sırasında hata: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    except Exception as e:
        print(f"\nVeri hazırlama sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        raise

# Alternatif basit loss fonksiyonu
def simple_custom_loss(y_true, y_pred):
    """
    Basitleştirilmiş loss fonksiyonu - sorun devam ederse bunu kullanabilirsiniz
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Temel BCE loss
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Long pozisyonlar için daha yüksek ağırlık
    weights = tf.ones_like(y_true) * 1.5
    
    return tf.reduce_mean(bce * tf.reduce_mean(weights, axis=1))

def custom_loss(y_true, y_pred):
    """
    Short pozisyonlar için iyileştirilmiş loss fonksiyonu
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # BCE hesaplama
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Short pozisyonlar için daha yüksek ağırlık
    short_weight = 5.0  # Arttırıldı
    long_weight = 3.5   # Aynı kaldı
    
    # Her sınıf için ayrı ağırlıklar
    weights = tf.ones_like(y_true)
    weights = weights + (long_weight * y_true[:, 0:1]) + (short_weight * y_true[:, 1:2])
    
    # Focal loss bileşeni - short için daha agresif
    gamma = 4.0  # Arttırıldı
    alpha = 0.85  # Arttırıldı
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    focal_weight = alpha * tf.pow(1 - p_t, gamma)
    
    # Short tahminleri için ek ceza
    short_uncertainty = tf.abs(y_pred[:, 1:2] - 0.5) * 2.0
    short_penalty = tf.reduce_mean(short_uncertainty) * 0.1
    
    # Final loss hesaplama
    weighted_loss = bce * tf.reduce_mean(weights, axis=1) * tf.reduce_mean(focal_weight, axis=1)
    
    return tf.reduce_mean(weighted_loss) + short_penalty

def create_model(input_shape):
    model = Sequential([
        # Giriş LSTM katmanı
        LSTM(256, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(0.001)),  # L2 regularization arttırıldı
        BatchNormalization(),
        Dropout(0.4),  # Dropout arttırıldı
        
        # Orta LSTM katmanı
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Son LSTM katmanı
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        
        # Dense katmanları güçlendirildi
        Dense(128, activation='relu'),  # Yeni katman eklendi
        BatchNormalization(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # Arttırıldı
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def prepare_balanced_data(X_windows, y_long_windows, y_short_windows):
    print("\nVeri dengeleme başlıyor...")
    
    # Long pozisyonlar için daha fazla örnek al
    n_long_pos = np.sum(y_long_windows == 1)
    n_long_neg = np.sum(y_long_windows == 0)
    
    # Hedef örnek sayısını artır
    target_samples = min(150000, n_long_pos * 4)  # Arttırıldı
    
    # Long pozitif örnekleri çoğalt
    long_pos_idx = np.where(y_long_windows == 1)[0]
    long_neg_idx = np.where(y_long_windows == 0)[0]
    
    # Pozitif örnekleri daha fazla örnekle
    pos_samples = np.random.choice(long_pos_idx, 
                                 size=target_samples // 2,
                                 replace=True)
    neg_samples = np.random.choice(long_neg_idx,
                                 size=target_samples // 2,
                                 replace=False)
    
    selected_indices = np.concatenate([pos_samples, neg_samples])
    np.random.shuffle(selected_indices)
    
    return (X_windows[selected_indices],
            y_long_windows[selected_indices],
            y_short_windows[selected_indices])

def predict_positions(predictions, threshold_long=0.45, threshold_short=0.15):
    """
    Daha kesin tahminler için threshold'ları ayarla
    """
    # Kararsız bölgeyi tanımla (0.4-0.6 arası)
    uncertain_mask = (predictions > 0.4) & (predictions < 0.6)
    
    # Kararsız tahminleri filtrele
    long_signals = (predictions[:, 0] > threshold_long) & (~uncertain_mask[:, 0])
    short_signals = (predictions[:, 1] < threshold_short) & (~uncertain_mask[:, 1])
    
    return long_signals, short_signals

def compile_and_train_model(model, X_train, y_train, X_test, y_test):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,  # Arttırıldı
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=1.0
    )
    
    # Learning rate scheduler ekle
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Early stopping güncellendi
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Sınıf ağırlıkları güncellendi
    class_weights = {
        0: 1.0,
        1: 3.0  # Arttırıldı
    }
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,  # Arttırıldı
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weights
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Detaylı model değerlendirme
    """
    print("\nModel Değerlendirmesi Başlıyor...")
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Long pozisyonları için metrikler
    print("\nLong Pozisyonu Detaylı Analiz:")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test[:, 0], y_pred_classes[:, 0]))
    
    print("\nConfusion Matrix:")
    cm_long = confusion_matrix(y_test[:, 0], y_pred_classes[:, 0])
    print(cm_long)
    
    # Short pozisyonları için metrikler
    print("\nShort Pozisyonu Detaylı Analiz:")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test[:, 1], y_pred_classes[:, 1]))
    
    print("\nConfusion Matrix:")
    cm_short = confusion_matrix(y_test[:, 1], y_pred_classes[:, 1])
    print(cm_short)
    
    # ROC eğrileri
    plot_roc_curves(y_test, y_pred)
    
    return {
        'long_metrics': classification_report(y_test[:, 0], y_pred_classes[:, 0], output_dict=True),
        'short_metrics': classification_report(y_test[:, 1], y_pred_classes[:, 1], output_dict=True),
        'predictions': y_pred
    }

def predict_in_batches(model, X, batch_size=128):
    """
    Büyük veri setleri için batch'ler halinde tahmin yapar
    
    Parameters:
    -----------
    model : keras.Model
        Eğitilmiş model
    X : numpy.ndarray
        Tahmin yapılacak veri
    batch_size : int
        Batch boyutu
        
    Returns:
    --------
    numpy.ndarray
        Tahminler
    """
    print("\nTahminler yapılıyor...")
    predictions = []
    
    # Toplam batch sayısını hesapla
    total_batches = int(np.ceil(len(X) / batch_size))
    
    for i in range(0, len(X), batch_size):
        # Batch'i al
        batch = X[i:i + batch_size]
        
        # Tahmin yap
        batch_predictions = model.predict(batch, verbose=0)
        predictions.append(batch_predictions)
        
        # İlerlemeyi göster
        if (i // batch_size) % 80 == 0:
            print(f"İşlenen batch: {i // batch_size}/{total_batches}")
            print(f"Bellek kullanımı: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Tüm tahminleri birleştir
    predictions = np.vstack(predictions)
    
    print(f"\nTahmin şekli: {predictions.shape}")
    return predictions

def plot_training_metrics(history):
    """Eğitim metriklerini görselleştir"""
    plt.figure(figsize=(15, 5))
    
    # Loss grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'))
    plt.close()

def plot_roc_curves(y_test, y_pred):
    """ROC eğrilerini çiz"""
    plt.figure(figsize=(15, 5))
    
    # Long pozisyonları için ROC
    fpr_long, tpr_long, _ = roc_curve(y_test[:, 0], y_pred[:, 0])
    roc_auc_long = auc(fpr_long, tpr_long)
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr_long, tpr_long, label=f'Long (AUC = {roc_auc_long:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Long ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    
    # Short pozisyonları için ROC
    fpr_short, tpr_short, _ = roc_curve(y_test[:, 1], y_pred[:, 1])
    roc_auc_short = auc(fpr_short, tpr_short)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr_short, tpr_short, label=f'Short (AUC = {roc_auc_short:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Short ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'))
    plt.close()

def print_metrics_summary(y_test, y_pred):
    """Metriklerin özetini yazdır"""
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    print("\nLong Pozisyonları için Metrikler:")
    print("----------------------------------")
    print(f"Accuracy: {np.mean(y_test[:, 0] == y_pred_classes[:, 0]):.4f}")
    print(f"Precision: {np.sum((y_test[:, 0] == 1) & (y_pred_classes[:, 0] == 1)) / np.sum(y_pred_classes[:, 0] == 1):.4f}")
    print(f"Recall: {np.sum((y_test[:, 0] == 1) & (y_pred_classes[:, 0] == 1)) / np.sum(y_test[:, 0] == 1):.4f}")
    
    print("\nShort Pozisyonları için Metrikler:")
    print("----------------------------------")
    print(f"Accuracy: {np.mean(y_test[:, 1] == y_pred_classes[:, 1]):.4f}")
    print(f"Precision: {np.sum((y_test[:, 1] == 1) & (y_pred_classes[:, 1] == 1)) / (np.sum(y_pred_classes[:, 1] == 1) + 1e-10):.4f}")
    print(f"Recall: {np.sum((y_test[:, 1] == 1) & (y_pred_classes[:, 1] == 1)) / (np.sum(y_test[:, 1] == 1) + 1e-10):.4f}")

# Model eğitimi bölümünde try bloğunu güncelle
try:
    print("Veri yükleniyor...")
    all_data = []
    for pair in pairs:
        filename = f'{data_dir}/{pair}-5m-futures.feather'
        if os.path.exists(filename):
            print(f"Yükleniyor: {filename}")
            df = pd.read_feather(filename)
            df['pair'] = pair
            
            # Teknik göstergeleri hesapla
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macdsignal'] = macd.macd_signal()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Stochastic RSI
            stoch = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi'] = stoch.stochrsi_d()
            df['stoch_k'] = stoch.stochrsi_k()
            
            # Williams %R
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            
            # Momentum
            df['mom'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
            
            # ADX
            df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            
            # Trend gücü
            df['trend_strength'] = df['adx']
            
            # NaN değerleri temizle
            df = df.dropna()
            
            all_data.append(df)
        else:
            print(f"Dosya bulunamadı: {filename}")

    if not all_data:
        raise ValueError("Hiç veri dosyası bulunamadı!")

    print("\nTüm veriler birleştiriliyor...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # Veriyi hazırla
    X, y_long, y_short, features = prepare_data_for_training(combined_df)
    print(f"X shape: {X.shape}, y_long shape: {y_long.shape}, y_short shape: {y_short.shape}")
    
    # Veriyi train/test olarak böl
    print("\nVeri train/test olarak bölünüyor...")
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = np.column_stack([y_long[:train_size], y_short[:train_size]])
    y_test = np.column_stack([y_long[train_size:], y_short[train_size:]])
    
    print(f"Train şekilleri:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"Test şekilleri:")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Belleği temizle
    del X, y_long, y_short
    gc.collect()
    
    # Model oluştur
    print("\nModel oluşturuluyor...")
    input_shape = (WINDOW_SIZE, len(features))
    model = create_model(input_shape=input_shape)
    
    # Model özeti
    print("\nModel Mimarisi:")
    model.summary()
    
    # Model derleme ve eğitim
    print("\nModel derleniyor ve eğitiliyor...")
    history = compile_and_train_model(model, X_train, y_train, X_test, y_test)
    
    # Tahminleri yap
    print("Eğitim seti tahminleri yapılıyor...")
    train_predictions = predict_in_batches(model, X_train, batch_size=BATCH_SIZE)
    
    print("Test seti tahminleri yapılıyor...")
    test_predictions = predict_in_batches(model, X_test, batch_size=BATCH_SIZE)
    
    # Grafikleri çiz
    plt.style.use('default')
    
    # Sonuçları görselleştir
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Eğitim kaybı grafiği
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Birleştirilmiş Model - Kayıp', fontsize=14, pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    # Test tahminleri grafiği
    ax2.plot(y_test[:, 0], label='Gerçek Long Probability', alpha=0.7, linewidth=2)
    ax2.plot(y_test[:, 1], label='Gerçek Short Probability', alpha=0.7, linewidth=2)
    ax2.set_title('Birleştirilmiş Model - Test Tahminleri', fontsize=14, pad=20)
    ax2.set_xlabel('Zaman', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    # Grafik düzenini ayarla
    plt.tight_layout()

    # Grafikleri kaydet
    plt.savefig(os.path.join(plots_dir, f'combined_plot_{pairs[0]}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model performans metriklerini hesapla
    if test_predictions is not None:
        # Yeni detaylı değerlendirme
        evaluation_results = evaluate_model(model, X_test, y_test)
        
        print("\nModel Değerlendirme Sonuçları:")
        print(f"Long Pozisyonu Detaylı Analiz:")
        print(evaluation_results['long_metrics'])
        print(f"Short Pozisyonu Detaylı Analiz:")
        print(evaluation_results['short_metrics'])
        
        # Metrikleri görselleştir
        plot_training_metrics(history)
        plot_roc_curves(y_test, evaluation_results['predictions'])
        print_metrics_summary(y_test, evaluation_results['predictions'])

except Exception as e:
    print(f"\nHata oluştu: {e}")
    import traceback
    traceback.print_exc()
    model = None
    history = None

# History kontrolü ve model kaydetme
if history is not None and model is not None:
    try:
        # Modeli kaydet
        model.save(final_model_path)
        print("Model kaydedildi:", final_model_path)
        
        # History'yi kaydet
        history_file = os.path.join(models_dir, 'training_history.npy')
        np.save(history_file, history.history)
        print("Eğitim geçmişi kaydedildi:", history_file)
        
    except Exception as e:
        print(f"Model kaydetme sırasında hata oluştu: {e}")

# Belleği temizle
tf.keras.backend.clear_session()
gc.collect()

print("\nİşlem tamamlandı.")