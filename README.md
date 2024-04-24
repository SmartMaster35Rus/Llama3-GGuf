
### Характеристики ПК
Данный скрипт был разработан и протестирован на следующей конфигурации ПК:

|  Конфигурация  |  Детали спецификаций  |
|----------------|----------------------|
|  Процессор     |  Intel® Core™ i9-14900KF @3.20Ghz  |
|  Память        |  128 ГБ DDR4 4200 МГц (32+32+32+32)  |
|  Диск          |  M.2 PCIe SSD Samsung SSD 980 PRO 1000 ГБ  |
|  Диск          |  M.2 PCIe SSD XPG GAMMIX S11 Pro 1000 ГБ |
|  Дискретная графика  |  NVIDIA GeForce RTX 4090 24 ГБ  |
|  CudaToolkit   |  ver.12.3  |
|  OS   |  Windows 11 Pro |

### !!!Важно: На данный момент данный репозиторий находится в бета-тесте.!!!

## Установка и настройка

Для начала работы с Llama3 GGUF //beta-test, выполните следующие шаги:

Открываем папку в которой будет храниться проект Llama3 GGuf : 

```Shell
  git clone https://huggingface.co/Orenguteng/Lexi-Llama-3-8B-Uncensored
```

скачиваем 4 файла в нашу папку проекта :

  
  


### 1. Создание виртуального окружения

Для изоляции проекта рекомендуется создать новое виртуальное окружение. Выполните следующие команды в терминале:

```python
conda create -n Llama3 python=3.9 ##очень важно именно версия 3.9
conda activate Llama
```

### 2. Установка зависимостей

Установите необходимые библиотеки, выполнив следующие команды:

## Верссии tensorflow выше 2.10 не работают на Windows поэтому ставим версию ниже 2.11

```python
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install transformers
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121
pip install xformers --upgrade --index-url https://download.pytorch.org/whl/cu121
pip install jupyter notebook
python -m pip install "tensorflow<2.11"
```

### 3. Проверка настройки TensorFlow GPU

Для использования GPU в TensorFlow, убедитесь, что настройка выполнена корректно. Выполните следующую команду:

```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

если все настроено правильно вы получаете ответ :

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

```


---
license: llama3
tags:
- uncensored
- llama3
- instruct
- open
---

![image/png](https://cdn-uploads.huggingface.co/production/uploads/644ad182f434a6a63b18eee6/H6axm5mlmiOWnbIFvx_em.png)

This model is based on Llama-3-8b-Instruct, and is governed by [META LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://llama.meta.com/llama3/license/)

Lexi is uncensored, which makes the model compliant. You are advised to implement your own alignment layer before exposing the model as a service. It will be highly compliant with any requests, even unethical ones. 

You are responsible for any content you create using this model. Please use it responsibly.

Lexi is licensed according to Meta's Llama license. I grant permission for any use, including commercial, that falls within accordance with Meta's Llama-3 license.
