"""Русские названия классов COCO (на них обучены стандартные модели YOLO)."""

RU_LABELS = {
    "person": "Человек",
    "bicycle": "Велосипед",
    "car": "Автомобиль",
    "motorcycle": "Мотоцикл",
    "airplane": "Самолёт",
    "bus": "Автобус",
    "train": "Поезд",
    "truck": "Грузовик",
    "boat": "Лодка",
    "traffic light": "Светофор",
    "fire hydrant": "Пожарный гидрант",
    "stop sign": "Знак «Стоп»",
    "parking meter": "Паркомат",
    "bench": "Скамейка",
    "bird": "Птица",
    "cat": "Кошка",
    "dog": "Собака",
    "horse": "Лошадь",
    "sheep": "Овца",
    "cow": "Корова",
    "elephant": "Слон",
    "bear": "Медведь",
    "zebra": "Зебра",
    "giraffe": "Жираф",
    "backpack": "Рюкзак",
    "umbrella": "Зонт",
    "handbag": "Сумка",
    "tie": "Галстук",
    "suitcase": "Чемодан",
    "frisbee": "Фрисби",
    "skis": "Лыжи",
    "snowboard": "Сноуборд",
    "sports ball": "Мяч",
    "kite": "Воздушный змей",
    "baseball bat": "Бейсбольная бита",
    "baseball glove": "Бейсбольная перчатка",
    "skateboard": "Скейтборд",
    "surfboard": "Доска для серфинга",
    "tennis racket": "Теннисная ракетка",
    "bottle": "Бутылка",
    "wine glass": "Бокал",
    "cup": "Чашка",
    "fork": "Вилка",
    "knife": "Нож",
    "spoon": "Ложка",
    "bowl": "Миска",
    "banana": "Банан",
    "apple": "Яблоко",
    "sandwich": "Сэндвич",
    "orange": "Апельсин",
    "broccoli": "Брокколи",
    "carrot": "Морковь",
    "hot dog": "Хот-дог",
    "pizza": "Пицца",
    "donut": "Пончик",
    "cake": "Торт",
    "chair": "Стул",
    "couch": "Диван",
    "potted plant": "Растение в горшке",
    "bed": "Кровать",
    "dining table": "Стол",
    "toilet": "Унитаз",
    "tv": "Телевизор",
    "laptop": "Ноутбук",
    "mouse": "Мышь",
    "remote": "Пульт",
    "keyboard": "Клавиатура",
    "cell phone": "Телефон",
    "microwave": "Микроволновка",
    "oven": "Духовка",
    "toaster": "Тостер",
    "sink": "Раковина",
    "refrigerator": "Холодильник",
    "book": "Книга",
    "clock": "Часы",
    "vase": "Ваза",
    "scissors": "Ножницы",
    "teddy bear": "Плюшевый мишка",
    "hair drier": "Фен",
    "toothbrush": "Зубная щётка",
}


from labels_oiv7_ru import OIV7_RU  # noqa: E402

# Подписи для моделей с произвольными классами (YOLOE): запрос модели -> то, что ввёл пользователь
CUSTOM_LABELS: dict[str, str] = {}


def _clean(en: str) -> str:
    """«Bat (Animal)» -> «bat»: текст для текстового кодировщика модели."""
    return en.split(" (")[0].strip().lower()


# Обратный словарь русский -> английский (для запросов YOLOE). COCO важнее Open Images.
RU_TO_EN: dict[str, str] = {}
for _en, _ru in list(OIV7_RU.items()) + list(RU_LABELS.items()):
    RU_TO_EN[_ru.lower()] = _clean(_en)
EN_TO_RU: dict[str, str] = {_clean(en): ru for en, ru in list(OIV7_RU.items()) + list(RU_LABELS.items())}


def translate(name: str, lang: str = "ru") -> str:
    """Возвращает название класса на нужном языке (для неизвестных — как есть)."""
    if name in CUSTOM_LABELS:
        return CUSTOM_LABELS[name]
    if lang == "ru":
        return RU_LABELS.get(name) or OIV7_RU.get(name) or name
    return name


def has_cyrillic(text: str) -> bool:
    return any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)


def parse_custom(entry: str) -> dict:
    """Разбирает введённое пользователем название для YOLOE.

    «кружка»          -> подпись «кружка», запрос «mug» (перевод из словаря)
    «очки = glasses»  -> подпись «очки», запрос «glasses» (перевод указан вручную)
    «red car»         -> подпись «red car», запрос «red car»
    """
    entry = " ".join(str(entry).split())
    if "=" in entry:
        label, prompt = (p.strip() for p in entry.split("=", 1))
        return {"entry": entry, "label": label or prompt, "prompt": prompt.lower() or label.lower(),
                "known": bool(prompt) and not has_cyrillic(prompt)}
    if has_cyrillic(entry):
        en = RU_TO_EN.get(entry.lower())
        return {"entry": entry, "label": entry, "prompt": en or entry.lower(), "known": en is not None}
    return {"entry": entry, "label": EN_TO_RU.get(entry.lower(), entry), "prompt": entry.lower(), "known": True}
