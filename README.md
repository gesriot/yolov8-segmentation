# YOLOv8 segmentation with OpenCV

Сегментация изображений на C++23 с OpenCV 4 и моделями YOLOv8-seg:
библиотека + CLI, сборка через пресеты CMake, тесты через CTest.

Конвейер инференса: letterbox → blob → forward → декодер → маски.
Модель отдаёт тензор детекций `[1, 116, 8400]` и прототипы масок
`[1, 32, 160, 160]`; декодер транспонирует тензор, фильтрует строки по порогу
уверенности, применяет `cv::dnn::NMSBoxes` и восстанавливает маску для каждого
бокса из прототипов (подробности форматов – в
[`models/README.md`](models/README.md)). Результат – изображение с боксами и
масками и JSON с детекциями.

## Структура

- `cpp/` – библиотека (`cpp/src`, `cpp/include`) и CLI (`cpp/app`);
- `shared/` – список классов COCO и тестовое изображение;
- `models/yolov8/` – локальные ONNX-экспорты (игнорируются Git);
- `YOLOv8/` – локальные `.pt`-чекпоинты Ultralytics (игнорируются Git);
- `scripts/` – сборка OpenCV, загрузка ONNX Runtime, сборка проекта и
  экспорт моделей;
- `third_party/` – игнорируется; локальная сборка OpenCV
  (см. [`docs/opencv-windows.md`](docs/opencv-windows.md)) и
  распакованный ONNX Runtime.

## Зависимости

- CMake >= 3.24, Ninja (желательно), MSVC с поддержкой C++23
  (Visual Studio C++ desktop workload);
- OpenCV >= 4.7 – собирается локально скриптом (пиновано `4.14.0`);
- ONNX Runtime – скачивается скриптом
  `scripts\get_onnxruntime_windows.ps1` (пиновано `1.27.1`); без него
  проект собирается только с движком OpenCV DNN;
- Python 3.12 – для экспорта моделей в ONNX.

## Модель

OpenCV DNN работает с ONNX, а не с `.pt`-чекпоинтами. Для сегментации нужны
именно `yolov8*-seg` (без суффикса `-seg` – это detection-модели). Экспорт
через виртуальное окружение в корне проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install ultralytics onnx onnxslim
.\.venv\Scripts\python scripts\export_yolov8_seg.py
```

Скрипт берёт чекпоинты `yolov8{n,s,m,l,x}-seg.pt` из `YOLOv8/` (недостающие
скачивает туда сам) и кладёт ONNX-экспорты в `models/yolov8/`. Отдельные
размеры: `--sizes n s`.

## Сборка и проверка

```powershell
.\scripts\build_windows.ps1
```

Скрипт последовательно:

- собирает OpenCV 4.14.0 в `third_party\opencv\4.14.0\install`, если его
  ещё нет (`scripts\build_opencv_windows.ps1`, см.
  [`docs/opencv-windows.md`](docs/opencv-windows.md));
- активирует окружение MSVC x64 и добавляет DLL OpenCV в `PATH`;
- конфигурирует и собирает проект по пресету `windows-release`;
- запускает CTest: тесты `yolov8{n,s,m}-seg-bus` выполняются для тех моделей,
  которые есть в `models/yolov8/`.

Опции: `-SkipOpenCV` (не проверять/не собирать OpenCV), `-SkipTests`,
`-NinjaPath C:\tools\ninja.exe`. Вручную – из терминала с окружением MSVC:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release
ctest --preset windows-release
```

## Запуск

```powershell
build\windows-release\cpp\yolov8-seg.exe `
  --model models\yolov8\yolov8n-seg.onnx `
  --input shared\testdata\bus.jpg `
  --classes shared\classes\coco80.txt `
  --output outputs\yolov8\bus.png `
  --json outputs\yolov8\bus.json
```

Любая из моделей `n`, `s`, `m`, `l`, `x` работает через один декодер.
Инференс выполняет ONNX Runtime (`--engine ort`, по умолчанию) или
OpenCV DNN (`--engine opencv`); оба движка дают идентичные детекции.
Дополнительные параметры: `--conf 0.25`, `--nms 0.45`, `--mask-threshold 0.5`,
`--warmup`, `--iterations` (при нескольких итерациях печатается min/медиана
времени инференса). Без `--classes` имена классов в JSON остаются
пустыми, а подписи на изображении становятся `class-<id>`.

Для запуска exe вне `build_windows.ps1` каталог с DLL OpenCV
(`third_party\opencv\4.14.0\install\bin`) должен быть в `PATH`
(CTest прописывает его в окружение тестов сам).

## Лицензии

Код проекта – MIT. Модели Ultralytics – AGPL-3.0 (если нет отдельной
Enterprise-лицензии).
