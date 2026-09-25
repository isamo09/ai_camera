"""Простейший локальный переводчик русский -> английский для своих названий YOLOE.

Работает без интернета и без больших моделей: перевод по словам через словарь.
Русские окончания (падежи, род, число) отбрасываются, поэтому «зелёная», «зелёного»,
«зелёные» находят одно слово green. Названия из нескольких слов («мобильный телефон»)
сначала ищутся целиком. Английский порядок «прилагательное + существительное» совпадает
с русским, поэтому для коротких названий этого достаточно: «красная машина» -> «red car»,
«кружка с котом» -> «mug with cat».
"""
from __future__ import annotations

import re

# Окончания, которые отбрасываются при поиске (длинные раньше коротких)
_ENDINGS = sorted("""
ыми ими ого его ому ему ая яя ое ее ые ие ый ий ой ей ую юю ым им ых их
ами ями ах ях ов ев ом ем ам ям
а я ы и у ю е о ь й
""".split(), key=len, reverse=True)

# Словарь для фраз: цвета, размеры, материалы, состояния, предлоги и частые предметы,
# которых нет среди классов COCO и Open Images.
WORDS = {
    # цвета
    "красный": "red", "синий": "blue", "голубой": "light blue", "зелёный": "green", "жёлтый": "yellow",
    "оранжевый": "orange", "фиолетовый": "purple", "розовый": "pink", "белый": "white", "чёрный": "black",
    "серый": "gray", "коричневый": "brown", "бежевый": "beige", "золотой": "golden", "серебряный": "silver",
    "бирюзовый": "turquoise", "прозрачный": "transparent", "тёмный": "dark", "светлый": "light",
    "яркий": "bright", "разноцветный": "colorful", "полосатый": "striped", "клетчатый": "checkered",
    # размер и форма
    "большой": "big", "маленький": "small", "огромный": "huge", "крошечный": "tiny", "высокий": "tall",
    "низкий": "low", "длинный": "long", "короткий": "short", "широкий": "wide", "узкий": "narrow",
    "толстый": "thick", "тонкий": "thin", "круглый": "round", "квадратный": "square",
    "прямоугольный": "rectangular", "плоский": "flat",
    # материал
    "деревянный": "wooden", "металлический": "metal", "железный": "iron", "пластиковый": "plastic",
    "стеклянный": "glass", "бумажный": "paper", "картонный": "cardboard", "кожаный": "leather",
    "каменный": "stone", "тканевый": "fabric", "резиновый": "rubber", "шерстяной": "wool",
    "керамический": "ceramic", "мягкий": "soft",
    # состояние
    "открытый": "open", "закрытый": "closed", "пустой": "empty", "полный": "full", "сломанный": "broken",
    "новый": "new", "старый": "old", "мокрый": "wet", "грязный": "dirty", "чистый": "clean",
    "включённый": "turned on", "выключенный": "turned off", "горящий": "burning", "спящий": "sleeping",
    "бегущий": "running", "сидящий": "sitting", "стоящий": "standing", "лежащий": "lying",
    "детский": "child's", "мужской": "men's", "женский": "women's", "кухонный": "kitchen",
    "электрический": "electric", "беспроводной": "wireless", "игрушечный": "toy", "живой": "live",
    # числа
    "один": "one", "два": "two", "две": "two", "три": "three", "несколько": "several", "много": "many",
    # предлоги и союзы
    "с": "with", "со": "with", "без": "without", "на": "on", "в": "in", "во": "in", "под": "under",
    "над": "above", "у": "near", "около": "near", "рядом": "near", "возле": "near", "для": "for",
    "из": "from", "от": "from", "до": "to", "по": "along", "за": "behind", "перед": "in front of", "между": "between", "и": "and", "или": "or",
    # люди и части тела
    "ребёнок": "child", "малыш": "baby", "младенец": "baby", "дети": "children", "люди": "people",
    "парень": "guy", "мужчина": "man", "женщина": "woman", "девушка": "young woman", "старик": "old man",
    "бабушка": "grandmother", "дедушка": "grandfather", "палец": "finger", "ладонь": "palm",
    "плечо": "shoulder", "колено": "knee", "шея": "neck", "спина": "back", "губы": "lips", "зуб": "tooth",
    "улыбка": "smile", "татуировка": "tattoo",
    # одежда и аксессуары
    "футболка": "t-shirt", "свитер": "sweater", "толстовка": "hoodie", "штаны": "pants", "кепка": "cap",
    "шапка": "beanie", "каска": "hard hat", "маска": "face mask", "кроссовка": "sneaker",
    "кроссовки": "sneakers", "тапок": "slipper", "тапки": "slippers", "носки": "socks", "варежка": "mitten",
    "браслет": "bracelet", "кольцо": "ring", "цепочка": "chain", "кошелёк": "wallet", "бейдж": "badge",
    "зонтик": "umbrella", "пакет": "bag", "мешок": "sack",
    # техника и электроника
    "провод": "cable", "кабель": "cable", "зарядка": "charger", "зарядное": "charger", "колонка": "speaker",
    "динамик": "speaker", "розетка": "power socket", "вилка": "plug", "удлинитель": "extension cord",
    "флешка": "usb flash drive", "мышка": "computer mouse", "экран": "screen", "джойстик": "gamepad",
    "геймпад": "gamepad", "приставка": "game console", "роутер": "router", "камера": "camera",
    "вебкамера": "webcam", "смартфон": "smartphone", "айфон": "iphone", "батарейка": "battery",
    "аккумулятор": "battery", "лампочка": "light bulb", "фонарик": "flashlight", "пылесос": "vacuum cleaner",
    "утюг": "iron", "дрон": "drone", "клавиша": "key", "кнопка": "button", "системник": "computer case",
    "компьютер": "computer", "процессор": "processor", "видеокарта": "graphics card",
    # дом, кухня, мебель
    "стакан": "glass", "банка": "jar", "коробка": "box", "ключ": "key", "ключи": "keys", "замок": "lock",
    "стол": "table", "кастрюля": "pot", "сковородка": "frying pan", "тарелка": "plate", "чашка": "cup", "кружка": "mug",
    "ложка": "spoon", "нож": "knife", "салфетка": "napkin", "губка": "sponge", "мыло": "soap",
    "шампунь": "shampoo", "зубная": "tooth", "щётка": "brush", "расчёска": "comb", "кресло": "armchair",
    "шкаф": "cabinet", "тумбочка": "nightstand", "полка": "shelf", "ковёр": "carpet", "одеяло": "blanket",
    "плед": "blanket", "штора": "curtain", "подоконник": "windowsill", "батарея": "radiator", "горшок": "pot",
    "цветок": "flower", "цветы": "flowers", "листья": "leaves", "деревья": "trees", "стулья": "chairs",
    "котята": "kittens", "глаза": "eyes", "уши": "ears", "растение": "plant", "кактус": "cactus", "пол": "floor", "потолок": "ceiling",
    "стена": "wall", "дверь": "door", "окно": "window", "ручка": "pen", "карандаш": "pencil",
    "маркер": "marker", "фломастер": "marker", "тетрадь": "notebook", "блокнот": "notepad",
    "лист": "sheet of paper", "бумага": "paper", "документ": "document", "папка": "folder",
    "карта": "card", "картина": "painting", "фотография": "photo", "фото": "photo", "мусор": "trash",
    "ведро": "bucket", "корзина": "basket", "ящик": "box", "игрушка": "toy", "кубик": "cube",
    "конструктор": "lego", "пазл": "puzzle", "мяч": "ball", "шарик": "balloon", "свечка": "candle",
    "зажигалка": "lighter", "сигарета": "cigarette", "пепельница": "ashtray", "бутылка": "bottle",
    "таблетка": "pill", "лекарство": "medicine", "градусник": "thermometer",
    # еда
    "хлеб": "bread", "батон": "loaf", "сыр": "cheese", "колбаса": "sausage", "сосиска": "sausage",
    "мясо": "meat", "курица": "chicken", "рыба": "fish", "яйцо": "egg", "шоколад": "chocolate",
    "печенька": "cookie", "чипсы": "chips", "бутерброд": "sandwich", "суп": "soup", "каша": "porridge",
    "огурец": "cucumber", "помидор": "tomato", "лук": "onion", "чеснок": "garlic", "перец": "pepper",
    "ягода": "berry", "вишня": "cherry", "малина": "raspberry", "мандарин": "tangerine", "орех": "nut",
    "вода": "water", "чай": "tea", "кофе": "coffee", "молоко": "milk", "сок": "juice", "пиво": "beer",
    # животные
    "кот": "cat", "котёнок": "kitten", "кошка": "cat", "щенок": "puppy", "пёс": "dog", "собака": "dog",
    "рыбка": "fish", "мышь": "mouse", "крыса": "rat", "голубь": "pigeon", "ворона": "crow",
    "курица": "chicken", "петух": "rooster", "муха": "fly", "комар": "mosquito", "таракан": "cockroach",
    # транспорт и улица
    "машина": "car", "автомобиль": "car", "самокат": "scooter", "электросамокат": "electric scooter",
    "мопед": "moped", "трамвай": "tram", "троллейбус": "trolleybus", "метро": "subway", "дорога": "road",
    "тротуар": "sidewalk", "забор": "fence", "столб": "pole", "знак": "sign", "вывеска": "signboard",
    "номер": "license plate", "колесо": "wheel", "руль": "steering wheel", "шлагбаум": "barrier",
    "дом": "house", "подъезд": "entrance", "магазин": "shop", "парк": "park", "дерево": "tree",
    "куст": "bush", "трава": "grass", "снег": "snow", "лужа": "puddle", "небо": "sky", "облако": "cloud",
    # инструменты
    "отвёртка": "screwdriver", "молоток": "hammer", "дрель": "drill", "пила": "saw", "плоскогубцы": "pliers",
    "рулетка": "tape measure", "гвоздь": "nail", "шуруп": "screw", "болт": "bolt", "гайка": "nut",
}


def _norm(word: str) -> str:
    return word.lower().replace("ё", "е")


def _stem(word: str) -> str:
    w = _norm(word)
    if len(w) <= 3:  # короткие слова (кот, пёс, чай) не режем
        return w
    for end in _ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 3:
            return w[: -len(end)]
    return w


def _clean_en(en: str) -> str:
    return en.split(" (")[0].strip().lower()


_EXACT: dict[str, str] = {}                       # точная форма слова -> перевод
_STEMS: dict[str, tuple[int, int, str]] = {}      # основа -> (приоритет, длина исходного слова, перевод)
_PHRASE_INDEX: dict[tuple[str, ...], str] = {}


def _add_stem(stem: str, word: str, en: str, priority: int) -> None:
    """При совпадении основ побеждает словарь фраз, затем более короткая (начальная) форма:
    «ключ» и «ключи» дают основу «ключ» — для неё останется перевод слова «ключ»."""
    rank = (priority, -len(word))
    old = _STEMS.get(stem)
    if old is None or rank > (old[0], old[1]):
        _STEMS[stem] = (priority, -len(word), en)


def _add(ru: str, en: str, priority: int = 0) -> None:
    words = re.findall(r"[а-яё]+", _norm(ru))
    if not words:
        return
    key = tuple(_stem(w) for w in words)
    if len(key) == 1:
        w = _norm(ru)
        if priority or w not in _EXACT:
            _EXACT[w] = en
        _add_stem(key[0], w, en, priority)
        if len(w) > 4 and w.endswith(("ок", "ек", "ец")):  # беглая гласная: горшок -> горшке, щенок -> щенки
            _add_stem(w[:-2] + w[-1], w, en, priority)
    elif priority or key not in _PHRASE_INDEX:
        _PHRASE_INDEX[key] = en


def build(class_names: dict[str, str]) -> None:
    """Добавляет в словарь названия классов моделей (русское -> английское)."""
    for ru, en in class_names.items():
        _add(ru, _clean_en(en))


for _ru, _en in WORDS.items():  # словарь фраз важнее названий классов
    _add(_ru, _en, priority=1)


def translate(text: str) -> tuple[str, list[str]]:
    """Переводит фразу. Возвращает (английский текст, список непереведённых слов)."""
    tokens = re.findall(r"[a-zA-Z0-9\-]+|[а-яА-ЯёЁ]+", text)
    stems = [_stem(t) if re.match(r"[а-яА-ЯёЁ]", t) else None for t in tokens]
    out, missing, i = [], [], 0
    while i < len(tokens):
        if stems[i] is None:  # латиница и числа — как есть
            out.append(tokens[i].lower())
            i += 1
            continue
        for n in (3, 2):  # сначала названия из нескольких слов
            key = tuple(stems[i:i + n])
            if len(key) == n and None not in key and key in _PHRASE_INDEX:
                out.append(_PHRASE_INDEX[key])
                i += n
                break
        else:
            stem = _STEMS.get(stems[i])
            en = _EXACT.get(_norm(tokens[i])) or (stem[2] if stem else None)
            if en:
                out.append(en)
            else:
                out.append(tokens[i].lower())
                missing.append(tokens[i])
            i += 1
    return " ".join(out), missing
