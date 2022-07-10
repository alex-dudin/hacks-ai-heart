# Решение задачи "Сердце" (hacks-ai.ru)

## Настройка окружения

Необходимо использовать Python 3.9 и установить пакеты из файла requirements.txt

Пример с использование conda:
```
conda create --name yaroslavl python 3.9
conda activate yaroslavl
pip install -r requirements.txt
```

## Обучение моделей и получение предсказаний

Для обучение моделей и получения предсказаний используется скрипт `fit_predict.py`.

Необходимо вызвать его 2 раза, для `lightgbm` и для `tabnet`.

Пример:
```
python fit_predict.py -i {input-data-dir} -o tabnet.csv -m tabnet --log-path tabnet.log
python fit_predict.py -i {input-data-dir} -o lightgbm.csv -m lightgbm --log-path lightgbm.log
```

## Получение финального решения

Для получения финального решения, необходимо усреднить предсказания от моделей `lightgbm` и `tabnet`.
Для этого используется скрипт `create_submission.py`.

Пример:
```
python create_submission.py -i lightgbm.csv -i tabnet.csv -o submission.csv --tresholds 0.47;0.045;0.105;0.09;0.09
```
