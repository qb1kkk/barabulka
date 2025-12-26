# LSTM для прогноза энергоэффективности облачной инфраструктуры

## Описание проекта

Этот проект посвящён построению и сравнительному анализу моделей прогнозирования метрики `energy_efficiency` для виртуальных машин в облачной инфраструктуре на основе временных рядов телеметрии. В качестве основной модели используется рекуррентная нейросеть LSTM, реализованная в Keras/TensorFlow. Дополнительно реализованы модели FCN, 1D-CNN, ARIMA и Exponential Smoothing для сравнения качества прогноза.

Модель обучается на временных рядах, сформированных из логов работы виртуальных машин (CPU, память, сеть, энергопотребление и др.), и прогнозирует будущие значения энергоэффективности. Проект включает полноценный пайплайн: от загрузки и предобработки данных до обучения, оценки качества и демонстрации работы через HTML-приложение.

## Структура репозитория

```text
.
├── LSTM_energy_efficiency_presentation.pptx   # Презентация для защиты (PPTX)
├── lstm_energy_efficiency_app.html            # HTML-приложение для демонстрации
├── vmCloud_data.csv                           # Датасет с логами VM
├── main.py / notebook.ipynb                   # Основной код (в вашем случае – ноутбук/скрипт)
└── README.md                                  # Описание проекта (этот файл)
```

> Примечание: имена файлов можно адаптировать под вашу фактическую структуру (например, `main.ipynb`, `lstm_model.ipynb`, `vmCloud_data.csv` и т.д.).

## Постановка задачи

**Цель:** построить модель, прогнозирующую метрику энергоэффективности `energy_efficiency` виртуальных машин в облачной инфраструктуре по историческим логам мониторинга.

**Задачи:**
- Сформировать датасет на основе логов VM (телеметрия CPU, память, сеть, энергопотребление и др.).
- Провести предобработку данных и нормализацию признаков.
- Построить нескольких моделей прогнозирования временного ряда:
  - LSTM (основная модель);
  - FCN (полносвязная сеть);
  - 1D-CNN (свёрточная сеть);
  - ARIMA;
  - Exponential Smoothing (Holt–Winters).
- Сравнить модели по метрикам MAE и MSE.
- Реализовать демонстрацию работы НС через HTML-приложение, принимающее на вход CSV с логами.

## Используемые технологии

- Язык: Python 3.x
- Библиотеки для работы с данными:
  - `pandas`, `numpy`
- Визуализация:
  - `matplotlib`, `seaborn`, `plotly`
- Машинное обучение и нейросети:
  - `tensorflow`, `keras`
  - `scikit-learn`
- Временные ряды:
  - `statsmodels` (ARIMA, ExponentialSmoothing)
- Веб-интерфейс для демонстрации:
  - Чистый HTML/CSS/JavaScript + Plotly

## Данные

### Источник и формат

- Входной файл: `vmCloud_data.csv`.
- Основные колонки:
  - `timestamp` — временная метка (дата/время наблюдения);
  - `cpu_usage` — загрузка CPU, %;
  - `memory_usage` — использование памяти, %;
  - `network_traffic` — сетевой трафик;
  - `power_consumption` — энергопотребление, Вт;
  - `num_executed_instructions` — количество выполненных инструкций;
  - `execution_time` — время выполнения задач;
  - `energy_efficiency` — целевая метрика энергоэффективности.

### Предобработка

1. Загрузка данных:
   ```python
   df = pd.read_csv(PARAMS["DATA_PATH"])
   ```

2. Обработка временной метки и сортировка:
   ```python
   df[PARAMS["TIME_COLUMN"]] = pd.to_datetime(df[PARAMS["TIME_COLUMN"]])
   df.sort_values(PARAMS["TIME_COLUMN"], inplace=True)
   ```

3. Работа с пропусками:
   ```python
   num_cols = [c for c in df.columns if df[c].dtype != "object" and c != PARAMS["TIME_COLUMN"]]
   cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != PARAMS["TIME_COLUMN"]]

   df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
   for c in cat_cols:
       df[c] = df[c].fillna("unknown")
   ```

4. Формирование поднабора признаков:
   ```python
   feature_cols = PARAMS["FEATURE_COLUMNS"]
   target_col = PARAMS["TARGET_COLUMN"]

   df_model = df[feature_cols + [target_col]].copy()
   ```

5. Масштабирование (MinMax):
   ```python
   scaler = MinMaxScaler()
   scaled_values = scaler.fit_transform(df_model.values)
   scaled_df = pd.DataFrame(scaled_values, columns=feature_cols + [target_col])
   ```

6. Формирование временных окон (последовательностей) для LSTM:
   ```python
   def create_sequences(data, seq_len, horizon, target_index):
       X, y = [], []
       for i in range(len(data) - seq_len - horizon + 1):
           X.append(data[i:i+seq_len])
           y.append(data[i+seq_len + horizon - 1, target_index])
       return np.array(X), np.array(y)

   X_all, y_all = create_sequences(
       scaled_df.values,
       PARAMS["SEQ_LEN"],
       PARAMS["HORIZON"],
       target_index
   )
   ```

7. Разбиение на обучающую и тестовую выборки:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X_all, y_all,
       test_size=PARAMS["TEST_SIZE"],
       shuffle=False
   )
   ```

## Архитектура моделей

### 1. LSTM (основная модель)

```python
model_lstm = Sequential([
    LSTM(PARAMS["LSTM_UNITS_1"], return_sequences=True,
         input_shape=(PARAMS["SEQ_LEN"], len(feature_cols)+1)),
    Dropout(PARAMS["DROPOUT_1"]),
    LSTM(PARAMS["LSTM_UNITS_2"]),
    Dropout(PARAMS["DROPOUT_2"]),
    Dense(PARAMS["DENSE_UNITS"], activation="relu"),
    Dense(1, activation="linear")
])
```

- **Тип:** рекуррентная нейросеть для временных рядов (sequence-to-one).
- **Вход:** окно длиной `SEQ_LEN = 16` по всем признакам.
- **Выход:** скалярное значение `energy_efficiency` на шаг вперёд.

**Компиляция и обучение:**

```python
opt = tf.keras.optimizers.Adam(learning_rate=PARAMS["LEARNING_RATE"])
model_lstm.compile(loss="mse", optimizer=opt, metrics=["mae"])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=PARAMS["PATIENCE"],
    restore_best_weights=True
)

history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=PARAMS["EPOCHS"],
    batch_size=PARAMS["BATCH_SIZE"],
    callbacks=[early_stop],
    verbose=1
)
```

### 2. FCN (полносвязная сеть)

```python
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

model_fcn = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_flat.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model_fcn.compile(loss="mse", optimizer="adam", metrics=["mae"])
```

### 3. 1D-CNN (свёрточная сеть)

```python
model_cnn = Sequential([
    Conv1D(32, kernel_size=3, activation="relu",
           input_shape=(PARAMS["SEQ_LEN"], len(feature_cols)+1)),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation="relu"),
    Dense(1, activation="linear")
])

model_cnn.compile(loss="mse", optimizer="adam", metrics=["mae"])
```

### 4. ARIMA

```python
series = df_model[target_col].values
train_size = int(len(series) * (1 - PARAMS["TEST_SIZE"]))
series_train, series_test = series[:train_size], series[train_size:]

model_arima = ARIMA(series_train, order=(2,1,2))
res_arima = model_arima.fit()
pred_arima = res_arima.forecast(steps=len(series_test))
```

### 5. Exponential Smoothing (Holt–Winters)

```python
model_es = ExponentialSmoothing(series_train, trend="add", seasonal=None)
res_es = model_es.fit()
pred_es = res_es.forecast(steps=len(series_test))
```

## Оценка качества моделей

Для оценки качества используются метрики `MAE` и `MSE`. После получения предсказаний выполняется обратное масштабирование целевой переменной, затем вычисляются метрики.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Пример для LSTM
y_pred_lstm_scaled = model_lstm.predict(X_test).flatten()
y_test_inv = inverse_scale_target_batch(y_test, template_row)
y_pred_lstm_inv = inverse_scale_target_batch(y_pred_lstm_scaled, template_row)

mae_lstm = mean_absolute_error(y_test_inv, y_pred_lstm_inv)
mse_lstm = mean_squared_error(y_test_inv, y_pred_lstm_inv)
```

Итоговая сводная таблица по моделям:

```python
metrics_df = pd.DataFrame([
    ["LSTM", mae_lstm, mse_lstm],
    ["FCN", mae_fcn, mse_fcn],
    ["1D-CNN", mae_cnn, mse_cnn],
    ["ARIMA", mae_arima, mse_arima],
    ["ExpSmoothing", mae_es, mse_es],
], columns=["Модель", "MAE", "MSE"])

print(metrics_df)
```

Также строится сравнительный bar-chart с помощью Plotly:

```python
fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(x=metrics_df["Модель"], y=metrics_df["MAE"], name="MAE"))
fig_comp.add_trace(go.Bar(x=metrics_df["Модель"], y=metrics_df["MSE"], name="MSE"))
fig_comp.update_layout(
    title="Сравнение 5 алгоритмов по MAE и MSE",
    barmode="group",
    template="plotly_white"
)
fig_comp.show()
```

## HTML-приложение для демонстрации

Проект включает автономное HTML-приложение `lstm_energy_efficiency_app.html`, которое позволяет демонстрировать работу модели в браузере.

### Возможности приложения

- Загрузка CSV-файла с логами виртуальных машин:
  ```text
  timestamp, cpu_usage, memory_usage, network_traffic,
  power_consumption, num_executed_instructions, execution_time, energy_efficiency
  ```
- Парсинг первых N строк (например, до 200) и извлечение столбца `energy_efficiency`.
- Расчёт базовой статистики: среднее, дисперсия, стандартное отклонение.
- Генерация демонстрационного прогноза (в данном HTML-коде — скользящая средняя для иллюстрации).
- Вычисление метрик MAE, MSE, RMSE между фактическими и "прогнозными" значениями.
- Построение интерактивного графика истинных и прогнозируемых значений с помощью Plotly.

### Структура интерфейса

- Блок загрузки файла (drag & drop или обычный выбор файла).
- Индикатор прогресса по шагам обработки (от чтения файла до построения графика).
- Таблица метрик (количество записей, MAE, MSE, RMSE, среднее, σ).
- График временного ряда (истинные vs прогноз).

HTML-файл может быть открыт локально двойным щелчком в любом современном браузере (Chrome, Firefox, Edge).

## Сохранение модели

Обученная LSTM-модель сохраняется в формате Keras `.h5`:

```python
model_lstm.save("lstm_energy_efficiency_model.h5")
```

Эту модель можно затем загрузить и использовать для инференса:

```python
from tensorflow.keras.models import load_model

model = load_model("lstm_energy_efficiency_model.h5")
```

## Как запустить проект

1. **Подготовить окружение**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scriptsctivate     # Windows

   pip install -r requirements.txt
   ```

2. **Разместить датасет**

   Убедиться, что файл `vmCloud_data.csv` находится в корне проекта или в пути, указанном в `PARAMS["DATA_PATH"]`.

3. **Запустить обучение (ноутбук или скрипт)**

   - Вариант 1: Jupyter/Colab ноутбук `lstm_model.ipynb`.
   - Вариант 2: Python-скрипт `main.py` (если код вынесен в скрипт).

4. **Запустить HTML-приложение**

   Открыть файл `lstm_energy_efficiency_app.html` в браузере и загрузить CSV.

## Дальнейшее развитие

- Добавление новых признаков (нагрузка по диску, температура, топология сети и т.д.).
- Использование attention-механизмов и/или Transformer-архитектур для временных рядов.
- Подбор гиперпараметров (Grid Search, Random Search, Bayesian Optimization, Optuna).
- Построение ансамблей из LSTM, 1D-CNN и статистических моделей.
- Обёртка модели в REST API (FastAPI/Flask) и деплой в Docker-контейнер.
- Интеграция с системами мониторинга и оркестрации (Prometheus, Grafana, Kubernetes).

## Лицензия

Укажите тип лицензии (например, MIT, Apache 2.0 или другую), если выкладываете проект в публичный репозиторий GitHub.

