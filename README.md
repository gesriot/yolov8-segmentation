# YOLOv8 segmentation
основано на https://github.com/UNeedCryDear/yolov8-opencv-onnxruntime-cpp

 Зависимость для opencv-dnn:
 > OpenCV >= 4.7.0<br>

Инструкция по сборке OpenCV из исходников: https://youtu.be/10Ni2gv4R7A

Собственно сборка проекта yolov8-segmentation:
- создаем директорию build в корневой директории проекта
- перейдим в директорию build
- выполняем команду cmake .. для генерации файлов сборки
- открываем yolov8-seg.sln в Visual Studio и собираем

Когда при сборке проекта (Release) возникнет ошибка: 
"не удается открыть входной файл ..\build\install\x64\vc16\lib.lib",
просто заходим в свойства проекта -> Компоновщик -> Ввод -> Доп.зависимости -> находим там эту ссылку и удаляем.

Потом возникнет ошибка рантайма.. кладем рядом с exe'ой (в .\build\Release) следующие файлы:
- opencv_core470.dll
- opencv_dnn470.dll
- opencv_highgui470.dll
- opencv_imgcodecs470.dll
- opencv_imgproc470.dll
- opencv_videoio470.dll

Здесь же создаем директории images и models для изображений и предобученных моделей соответственно.