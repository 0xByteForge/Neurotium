# Neurotium

```
FROM freqtradeorg/freqtrade:stable

USER root

# System paketlerini güncelle
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini yükle
RUN pip3 install --no-cache-dir \
    tensorflow==2.13.1 \
    keras==2.13.1 \
    pandas-ta \
    scikit-learn \
    joblib \
    psutil \
    numpy==1.24.3 \
    h5py==3.8.0

# Çalışma dizinini ayarla
WORKDIR /freqtrade

# Gerekli dizinleri oluştur
RUN mkdir -p /freqtrade/user_data/models
RUN mkdir -p /freqtrade/user_data/strategies

# Freqtrade kullanıcısına geri dön
USER freqtrade
```
