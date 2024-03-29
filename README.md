# FindYourFaceBot

Часто бывает, что нужно найти свои фотографии в альбоме, или фотографии с конкретными людьми, но тратить на это свое время слишком дорого, поэтому на помощь приходит этот бот.

В него можно загрузить фотографии, он автоматически распознает лица на них и с каждым из этих лиц можно найти все фотографии из загруженных в бота, также можно найти все фотографии в папке на Google Drive.

## Как запустить бота

- Зайдите в гугл-колаб https://colab.research.google.com/
- Откройте в нем ноутбук https://github.com/Sanches-wolf228/Diplom/blob/main/main.ipynb
- Запустите все ячейки, используя GPU

Потом можно перейти в него в телеграме https://t.me/FindYourFaceBot.

## Протестируйте его работу

Нажмите на `/help`, чтобы узнать подробнее про функционал сервиса.

В начале каждого нового теста перезапускайте бота, нажав на `/start`, чтобы нумерация началась заново.

### Testcase 1

- Загрузите фотографии с гугл-диска

`/load https://drive.google.com/drive/folders/15ZZydMgRQAIdUPktW5ZJof2ZxRkLFkht`

- Сравните, насколько лица похожи

`/compare 20 21`
`/compare 22 23`
`/compare 20 23`
`/compare 21 22`

- Разбейте лица на группы по людям, учитываются только группы из более, чем одного человека

`/group_by_face`

- Посмотрите на кластеры, на которые мы разбили лица

`/show 2 14`
`/show 3 13 15 17 18`
`/show 4 6 8 10 12 16 19 22 23`
`/show 5 7 9 11 20 21`

- Найдите все фотографии, на которых изображён отдельный человек

`/find_face 17`

- Найдите все фотографии, которые содержат сразу двух людей

`/find_party 4 5`
`/find_party 8 15`

- Отправьте боту собственные фотографии и поэкспериментируйте

### Testcase 2

- Загрузите фотографию по ссылке https://github.com/Sanches-wolf228/Diplom/blob/main/Collage.jpg

`/load https://raw.githubusercontent.com/Sanches-wolf228/Diplom/main/Collage.jpg

- Найдите фотографии этих людей на гугл-диске

`/find_gdrive 3 https://drive.google.com/drive/folders/1--3SNODXLaIPG37X-tQiu6xwnXFzB9Iq`

### Testcase 3

- Загрузите фотографию по ссылке https://github.com/Sanches-wolf228/Diplom/blob/main/4080HD.jpg

`/load https://raw.githubusercontent.com/Sanches-wolf228/Diplom/main/4080HD.jpg

- На фото все люди разные, убедимся в этом, сгруппировав по лицам

`/group_by_face`

- Что-то пошло не так, посмотрим на получившуюся группу

`/show 4 6 7`

`/compare 6 7`

- Видимо, из-за низкого разрешения и маленького размера лиц, точность распознавания понизилась, увеличим порог.

`/set_threshold 0.5`

`/group_by_face`

- Меняя параметры порога, можно добиваться лучшей точности распознавания
