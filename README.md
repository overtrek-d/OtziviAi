# Нейронная сеть для анализа отзывов / Neural network for review analysis

## Установка / Install
1. Установка библиотек / Installing libraries
> pip install torch torchvision torchaudio

> pip install numpy scikit-learn

> pip install PyYAML

> pip install flask

## Использование / Using
1. Обучение / Learning model
Подготовь датасет для обучения texts.yml
Prepare a dataset for training texts.yml
Формат / Format: [["text", 0], ["yetAnothertext", 1]]

> Run learn.py

2. Анализ / Analysis
Проверь параметры при использовании.
В переменную text напиши отзыв.
Check the settings when using.
Write a review in the text variable.

> Run pred.py

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
