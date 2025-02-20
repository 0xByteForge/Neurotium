import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
import gc
from tensorflow.keras.regularizers import l2

# GPU/CPU ayarları
print("Kullanılabilir GPU'lar:", tf.config.list_physical_devices('GPU'))

# GPU bellek ayarları
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        # Bellek limitini artır
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=6144)]  # 6GB
        )
        print("GPU bellek ayarları yapılandırıldı")
    except RuntimeError as e:
        print(e)

# CPU thread optimizasyonu
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Tüm normalize edilmiş CSV dosyalarını bul
csv_files = glob.glob('normalized_binance_*_5m_data.csv')

# Model ve eğitim parametreleri
BATCH_SIZE = 128  # 32'den 128'e çıkarıldı
NUM_WORKERS = 4  # Worker sayısını azalt
WINDOW_SIZE = 80  # Window boyutunu artır
STRIDE = 4  # Stride'ı azalt (daha fazla veri)
FUTURE_STEPS = 2

# Dizin yapılarını oluştur
log_dir = os.path.join("logs", "combined_model", datetime.now().strftime("%Y%m%d-%H%M%S"))
models_dir = "models"
plots_dir = "plots"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

model_checkpoint_path = os.path.join(models_dir, 'best_model_combined.h5')
final_model_path = os.path.join(models_dir, 'crypto_lstm_model_combined.h5')

# Veriyi parçalara bölen fonksiyon
def split_dataframe(df, chunk_size):
    num_chunks = len(df) // chunk_size
    if len(df) % chunk_size != 0:
        num_chunks += 1
    return np.array_split(df, num_chunks)

# Her sembol için veriyi işle
all_data = []
for csv_file in csv_files:
    symbol = csv_file.split('_')[2]
    print(f"\n{symbol} verisi işleniyor...")
    
    # Veriyi yükle
    df = pd.read_csv(csv_file)
    df['symbol'] = symbol
    
    # Veriyi 4 parçaya böl
    chunks = split_dataframe(df, len(df) // 6)  # 8 yerine 4 parça
    
    for i, chunk in enumerate(chunks):
        print(f"{symbol} - Parça {i+1}/{len(chunks)} işleniyor...")
        all_data.append(chunk)

print("\nTüm veriler birleştiriliyor...")
combined_df = pd.concat(all_data, ignore_index=True)

# Sembolleri one-hot encoding ile kodla
symbol_dummies = pd.get_dummies(combined_df['symbol'], prefix='symbol')
combined_df = pd.concat([combined_df, symbol_dummies], axis=1)

# Özellik sütunlarını belirle
feature_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 
                  'bb_upper', 'bb_lower', 'bb_middle', 'stoch_rsi', 'williams_r'] + \
                 list(symbol_dummies.columns)

# Veriyi dizilere dönüştür
X = combined_df[feature_columns].values.astype('float32')
y = combined_df['close'].values.astype('float32')

# Sliding Window fonksiyonu
def create_sliding_windows(data, window_size, stride, future_steps):
    windows = []
    targets = []
    for i in range(0, len(data) - window_size - future_steps + 1, stride):
        window = data[i:i + window_size]
        target = data[i + window_size:i + window_size + future_steps]
        windows.append(window)
        targets.append(target)
    return np.array(windows), np.array(targets)

print("Sliding windows oluşturuluyor...")
X_windows, _ = create_sliding_windows(X, WINDOW_SIZE, STRIDE, FUTURE_STEPS)
_, y_targets = create_sliding_windows(y, WINDOW_SIZE, STRIDE, FUTURE_STEPS)
X_seq = X_windows
y_seq = y_targets.reshape(-1, FUTURE_STEPS)

print("Eğitim ve test verileri ayrılıyor...")
train_size = int(len(X_seq) * 0.8)
X_train = X_seq[:train_size]
X_test = X_seq[train_size:]
y_train = y_seq[:train_size]
y_test = y_seq[train_size:]

# Belleği temizle
del combined_df, X, y, X_seq, y_seq, X_windows, y_targets
gc.collect()

# Her tahmin öncesi belleği temizle
tf.keras.backend.clear_session()
gc.collect()

# Model parametrelerini güncelle
model = Sequential([
    LSTM(384, return_sequences=True, 
         input_shape=(WINDOW_SIZE, len(feature_columns)),
         kernel_regularizer=l2(0.005),
         recurrent_regularizer=l2(0.005),
         activity_regularizer=l2(0.005)),
    BatchNormalization(momentum=0.95),
    Dropout(0.4),
    
    LSTM(128, return_sequences=True),
    BatchNormalization(momentum=0.95),
    Dropout(0.4),
    
    LSTM(64, return_sequences=True),
    BatchNormalization(momentum=0.95),
    Dropout(0.4),
    
    LSTM(32, return_sequences=False),
    BatchNormalization(momentum=0.95),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(FUTURE_STEPS)
])

# Optimizer ve eğitim parametrelerini güncelle
initial_learning_rate = 0.000005
optimizer = tf.keras.optimizers.Adam(
    learning_rate=initial_learning_rate,
    beta_1=0.95,
    beta_2=0.999,
    epsilon=1e-07
)

# Model derleme
model.compile(
    optimizer=optimizer,
    loss='huber',
    metrics=['mse', 'mae']
)

# Early stopping'i daha toleranslı yap
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=25,  # Daha uzun bekleme
    min_delta=0.00005  # Daha hassas iyileşme kontrolü
)

# Checkpoint ve model kaydetme ayarlarını güncelle
checkpoint_dir = os.path.join(models_dir, "checkpoints", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(checkpoint_dir, exist_ok=True)

# Checkpoint dosya yolu
checkpoint_path = os.path.join(checkpoint_dir, "model_checkpoint-{epoch:04d}-{val_loss:.4f}.h5")

# Model checkpoint callback'i
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,  # Tüm modeli kaydet
    save_best_only=True,  # Sadece en iyi modeli kaydet
    monitor='val_loss',
    mode='min',
    save_freq='epoch',
    verbose=1
)

# En iyi model callback'i - her çalıştırmada yeni bir dosya oluştur
best_model_path = os.path.join(models_dir, f'best_model_combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
best_model_callback = ModelCheckpoint(
    best_model_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Final model path'i de tarihle güncelle
final_model_path = os.path.join(models_dir, f'crypto_lstm_model_combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')

# Model özeti için sadece print kullan
model.summary()

# Loss değerlerini formatlamak için callback
class LossFormatter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEpoch {epoch+1}")
        for metric in ['loss', 'val_loss']:
            if metric in logs:
                print(f"{metric}: {logs[metric]:.8f}")  # 8 ondalık basamak göster

# Learning rate callback'i güncelle
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Callbacks listesini güncelle
callbacks = [
    early_stopping,
    checkpoint_callback,
    best_model_callback,
    LossFormatter(),
    lr_reducer  # Learning rate reducer'ı ekle
]

# Eğitim geçmişini yükle veya yeni oluştur
history_file = 'training_history.npy'
if os.path.exists(history_file):
    history_dict = np.load(history_file, allow_pickle=True).item()
    initial_history = {
        'loss': history_dict['loss'],
        'val_loss': history_dict['val_loss']
    }
else:
    initial_history = None

# Veri yükleme ve işleme stratejisini güncelle
def process_data_in_chunks(X_data, y_data, chunk_size=1000):
    num_samples = len(X_data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_indices = indices[start_idx:end_idx]
        
        yield X_data[chunk_indices], y_data[chunk_indices]

# Tahmin işlemini parçalara böl fonksiyonunu güncelle
def predict_in_batches(model, data, batch_size=32):
    predictions = []
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    print(f"Toplam {total_batches} batch işlenecek...")
    
    for i in range(0, len(data), batch_size):
        if i % (batch_size * 100) == 0:  # Her 100 batch'te bir durum raporu
            print(f"Batch {i//batch_size}/{total_batches} işleniyor...")
            
        batch = data[i:i + batch_size]
        # Belleği optimize et
        with tf.device('/GPU:0'):  # GPU kullanımını zorla
            pred = model.predict(batch, verbose=0, batch_size=batch_size)
        predictions.append(pred)
        
        # Daha az sıklıkta bellek temizleme
        if i % (batch_size * 1000) == 0:  # Her 1000 batch'te bir bellek temizle
            tf.keras.backend.clear_session()
            gc.collect()
    
    print("Tahminler birleştiriliyor...")
    final_predictions = np.concatenate(predictions)
    print(f"Tahmin şekli: {final_predictions.shape}")
    return final_predictions

# Model eğitimi bölümünde tahmin kısmını güncelle
try:
    # Eğitim öncesi belleği temizle
    tf.keras.backend.clear_session()
    gc.collect()
    
    history = model.fit(
        X_train, 
        y_train,
        validation_split=0.25,
        epochs=200,
        batch_size=128,  # 64'ten 128'e çıkarıldı
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=False,
        verbose=1,
        shuffle=False
    )
    
    # Eğitim sonrası belleği temizle
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Tahminleri yap
    print("Eğitim seti tahminleri yapılıyor...")
    train_predictions = predict_in_batches(model, X_train, batch_size=64)
    
    print("Test seti tahminleri yapılıyor...")
    test_predictions = predict_in_batches(model, X_test, batch_size=64)
    
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
    ax2.plot(y_test, label='Gerçek Değerler', alpha=0.7, linewidth=2)
    ax2.plot(test_predictions, label='Tahminler', alpha=0.7, linewidth=2)
    ax2.set_title('Birleştirilmiş Model - Test Tahminleri', fontsize=14, pad=20)
    ax2.set_xlabel('Zaman', fontsize=12)
    ax2.set_ylabel('Fiyat', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True)

    # Grafik düzenini ayarla
    plt.tight_layout()

    # Grafikleri kaydet
    plt.savefig(os.path.join(plots_dir, f'combined_plot_{symbol}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model performans metriklerini hesapla
    if test_predictions is not None:
        mse = tf.keras.losses.mean_squared_error(y_test, test_predictions).numpy().mean()
        mae = tf.keras.losses.mean_absolute_error(y_test, test_predictions).numpy().mean()
        
        # Performans metriklerini yazdır
        print("\nBirleştirilmiş Model Performansı:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Mean Absolute Error: {mae:.6f}")
    
except Exception as e:
    print(f"Hata oluştu: {e}")
    history = None
    train_predictions = None
    test_predictions = None
    tf.keras.backend.clear_session()
    gc.collect()

# History kontrolü
if history is not None:
    # Eğitim geçmişini kaydet
    if initial_history:
        combined_history = {
            'loss': initial_history['loss'] + history.history['loss'],
            'val_loss': initial_history['val_loss'] + history.history['val_loss']
        }
    else:
        combined_history = history.history
        
    np.save(history_file, combined_history)

# Modeli kaydet
model.save(final_model_path)
print("Birleştirilmiş model eğitimi tamamlandı ve kaydedildi.")
print(f"TensorBoard logları: {log_dir}")

# Model özeti ve grafik görselleştirmesi için
tf.keras.utils.plot_model(
    model,
    to_file=os.path.join(plots_dir, 'model_architecture_combined.png'),
    show_shapes=True,
    show_layer_names=True,
    expand_nested=True
)

print("\nTüm modeller eğitildi.")