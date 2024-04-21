# Локальный голосовой ввод
###### Речь в текст | STT | ASR | Speech to text

Эта утилита может в режиме реального времени локально распознавать речь с микрофона и вводить её в текстовое поле.

- Работает локально
- Распознаёт поток с микрофона на лету (паралелльно вводу), а не после записи
- Сразу вводит распознанный текст в поле

## Основной концепт
- Пишем звук с микрофона в буфер в отдельном потоке
- В основном потоке ждём заполнения буфера, подаём модели на распознавание
- Распознавание ведём кусочками, добавляем для затравки последние N (к примеру, 1000) символовов (с обрезкой по словам)
- После распознавания кусочка смотрим какие слова пересеклись (уже выведены)
- Определяем кандидатов для нового вывода, обрезаем последнее слово / сегмент, т.к. оно могло быть записано частично
- Запоминаем позицию последнего выведенного слова
- Отправляем кадидатов для вывода
- Добавляем в буфер ещё часть аудио, объединяем с частью, которую орезали от кандидатов на вывод, распознаём
- Повторяем, в конце распознаём и выводим всё что осталось

## Установка и настройка
- Скачать
- Установить зависимости `pip install -r requirements.txt`
- Открыть `main.py` и указать ваши параметры
	- language = 'ru' - целевой язык спискера
	- vad = False - исползование фильтра удаления молчания (требует тонкой подстройки, иначе удаляет знаки препинания)
	- size = 'large-v3' - доступная модель (также может потроваться изменить процессор в файле ASRProcessor.py, сейчас GPU)
	- SAMPLE_RATE = 16000 - частота дискретизации (16к - дефолт для whisper)
	- selected_device = 1 - виртуальный номер микрофона (0 - устойство по умолчанию, но может не работать)
Пока что все настройки надо делать в коде

## Использование
- Запустить
- Выбрать текстовое поле
- Нажать комбинацию hotkey `ctrl+alt+R` для запуска
- Говорить в выбранный микровон
- Нажать комбинацию hotkey `ctrl+alt+R` для остановки
- Дождаться, когда пропадёт индикатор записи, что означает завершение ввода текста

## Модели и производительность
Используется `faster-whisper` как лучшее по доступности и качеству решение на апрель 2024.
Mobile RTX 4070 на large-3 распознаёт на лету.
Альтернативные решение были не лучше, или требовали ёмкой настройки. Потенциально интересным решением является использование TensorRT (требует реализации backend через docker контейнер) или Whisper JAX (аналогично).

## Разработка
- `streaming_demo_bench.py` - скрипт для проверки возможности ASRProcessor (обёртки модели) распознавать на лету. Полезно для доработки логики распознавания.
- `ASRP_debug_demo.py` - фиктивная версия ASRProcessor, симулирующая работу и обеспечивающая диагнористический вывод для разработки интерфейса.

## ToDo
- [ ] Настройка распознавания (язык, модель, размер модели, микрофон)
- [ ] Графический интерфейс для смены настроек
- [ ] Работа в фонке
	- [ ] Иконка в трее
	- [ ] Автозапуск
	- [ ] Настройка автозапуска
- [ ] Уулчшить архитекутуру текущего распознавателя, улучшить код, оптимизировать

### Идеи улучшения ASR:
- Распознавать некоторую часть уже выведенного кусочка для лучшей связности, сличать 2 варианта нахлёста, корректировать вывод
- Добавить корректор смысла / связности поверх и исправлять уже выведенный текст