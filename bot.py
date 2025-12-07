import logging
import os
import sqlite3
from datetime import datetime
import re
import json
import emoji
import numpy as np
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import requests
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import atexit

# Загружаем переменные окружения
load_dotenv()

# Импортируем конфигурацию из config.py
try:
    from config import BOT_TOKEN, ADMIN_ID, NLTK_DATA_DIR
except ImportError:
    # Fallback на переменные окружения
    BOT_TOKEN = os.getenv('BOT_TOKEN')
    ADMIN_ID = int(os.getenv('ADMIN_ID', '6830411048'))
    NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIR', './nltk_data')

# === ВАЖНО: Настройка путей для Railway ===
def setup_railway_paths():
    """Настройка путей для работы на Railway"""
    
    # Создаем директории если их нет
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Настраиваем NLTK
    nltk.data.path.append(NLTK_DATA_DIR)
    
    # Путь к базе данных
    if os.getenv('RAILWAY_ENVIRONMENT'):
        # На Railway используем путь в /tmp для сохранения данных между рестартами
        db_path = '/tmp/bot_database.db'
        log_path = '/tmp/bot.log'
    else:
        # Локально используем текущую директорию
        db_path = 'bot_database.db'
        log_path = 'bot.log'
    
    return db_path, log_path

# Получаем пути
DB_PATH, LOG_PATH = setup_railway_paths()

# Скачиваем только ОСНОВНЫЕ данные NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Скачивание ОСНОВНЫХ данных NLTК...")
    nltk.download('punkt', quiet=True, download_dir=NLTK_DATA_DIR)
    nltk.download('stopwords', quiet=True, download_dir=NLTK_DATA_DIR)
    print("✅ Основные данные NLTK скачаны")

# Настройка логирования для Railway
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),  # Важно для просмотра логов в Railway
        logging.FileHandler(LOG_PATH)
    ]
)
logger = logging.getLogger(__name__)

# Логируем информацию о среде
logger.info("=" * 50)
logger.info(f"🚀 Запуск бота на Railway: {os.getenv('RAILWAY_ENVIRONMENT', 'Неизвестно')}")
logger.info(f"📁 База данных: {DB_PATH}")
logger.info(f"📁 NLTK данные: {NLTK_DATA_DIR}")
logger.info(f"✅ Токен присутствует: {'Да' if BOT_TOKEN else 'Нет'}")
logger.info(f"📏 Длина токена: {len(BOT_TOKEN) if BOT_TOKEN else 0}")
logger.info(f"👑 Админ ID: {ADMIN_ID}")
logger.info("=" * 50)

# Инициализация глобальных переменных для NLP
stop_words_ru = set(stopwords.words("russian"))
stop_words_en = set(stopwords.words("english"))
stemmer_ru = SnowballStemmer("russian")
stemmer_en = SnowballStemmer("english")

# Темы с подробными описаниями и ключевыми словами
DETAILED_TOPICS = {
    "Образование и Саморазвитие": {
        "keywords": ["учеба", "саморазвитие", "книги", "курсы", "образование", "знание", "развитие", "психология", "мышление", "обучение", "университет", "школа", "знания", "самосовершенствование", "мотивация", "цели", "успех"],
        "description": "Группа для тех, кто стремится к постоянному развитию, изучению нового и личностному росту.",
        "emoji": "📚"
    },
    "Наука и литература": {
        "keywords": ["наука", "литература", "книги", "авторы", "научные", "исследования", "научная", "фантастика", "классика", "поэзия", "проза", "литературные", "критика", "научпоп", "физика", "химия", "биология", "история"],
        "description": "Обсуждение научных открытий, литературных произведений и авторов, научной фантастики и классики.",
        "emoji": "🔬"
    },
    "Программирование": {
        "keywords": ["программирование", "код", "разработка", "python", "javascript", "веб", "мобильные", "приложения", "алгоритмы", "бэкенд", "фронтенд", "дата", "аналитика", "машинное", "обучение", "искусственный", "интеллект", "нейронные", "сети"],
        "description": "Группа для разработчиков, где обсуждаются языки программирования, фреймворки и технологии.",
        "emoji": "💻"
    },
    "Экономика и Бизнес": {
        "keywords": ["экономика", "бизнес", "финансы", "инвестиции", "стартап", "предпринимательство", "рынок", "деньги", "заработок", "доход", "прибыль", "капитал", "бизнесмен", "предприниматель", "трейдинг", "акции", "валюта", "криптовалюта", "форекс", "недвижимость"],
        "description": "Обсуждение экономических новостей, бизнес-идей, инвестиций и финансовых стратегий.",
        "emoji": "💰"
    },
    "Здоровье и медицина": {
        "keywords": ["здоровье", "медицина", "фитнес", "питание", "спорт", "йога", "лечение", "профилактика", "психическое", "здоровье", "диета", "витамины", "лекарства", "болезни", "врачи", "психология", "стресс", "сон", "релаксация", "оздоровление"],
        "description": "Группа о здоровье, фитнесе, правильном питании и медицинских аспектах.",
        "emoji": "💪"
    },
    "Искусство и музыка": {
        "keywords": ["искусство", "музыка", "творчество", "живопись", "рисование", "композиторы", "исполнители", "творческие", "художники", "графика", "скульpture", "архитектура", "классическая", "рок", "джаз", "поп", "эстрада", "инструменты", "гитара", "фортепиано"],
        "description": "Обсуждение искусства, музыки, творческих проектов и культурных событий.",
        "emoji": "🎨"
    },
    "Кулинария и рецепты": {
        "keywords": ["кулинария", "рецепты", "готовка", "еда", "блюда", "ингредиенты", "вкусно", "домашняя", "кухни", "выпечка", "кондитерское", "десерты", "салаты", "супы", "вторые", "блюда", "напитки", "кофе", "чай", "вино"],
        "description": "Группа для любителей готовить и обмениваться рецептами разных кухонь мира.",
        "emoji": "🍳"
    },
    "Путешествие и туризм": {
        "keywords": ["путешествие", "туризм", "страны", "город", "отдых", "отпуск", "достопримечательности", "экскурсии", "походы", "автомобильные", "туристические", "маршруты", "гостиницы", "отели", "авиабилеты", "визы", "пляж", "море", "горы", "природа", "экзотика", "бюджетные", "дорогие", "туристы"],
        "description": "Обсуждение путешествий, туристических маршрутов, стран и мест для отдыха.",
        "emoji": "✈️"
    },
    "Спорт": {
        "keywords": ["спорт", "фитнес", "тренеровка", "чемпионат", "матчи", "здоровье", "физическая", "активность", "командный", "футбол", "баскетбол", "волейбол", "теннис", "плавание", "бег", "велосипед", "единоборства", "бокс", "бои", "тренажерный", "зал", "диета", "питание"],
        "description": "Группа о спорте, физической активности и здоровом образе жизни.",
        "emoji": "⚽"
    },
    "Иное": {
        "keywords": ["разное", "другое", "всякое", "разное", "общее", "разные", "темы", "обсуждения", "общение", "флуд", "разговоры", "мемы", "юмор", "анекдоты", "интересное", "важное", "актуальное", "новости"],
        "description": "Группа для общения на разные темы, которые не вошли в другие категории.",
        "emoji": "🔄"
    }
}

# ID реальных Telegram групп
GROUP_IDS = {
    "Образование и Саморазвитие": "-1003433439121",
    "Наука и литература": "-1002820402117", 
    "Программирование": "-1003477061325",
    "Экономика и Бизнес": "-1003382139382",
    "Здоровье и медицина": "-1003305866632",
    "Искусство и музыка": "-1003378596165",
    "Кулинария и рецепты": "-1003210673239",
    "Путешествие и туризм": "-1003340734939",
    "Спорт": "-1003300649893",
    "Иное": "-1003307595772"
}

# Состояния для разговоров
MAIN_MENU, ASK_TOPIC, CHOOSE_TOPIC, JOIN_CHAT, SUPPORT = range(5)

# Глобальные переменные для кэширования
topic_vectors = None
vectorizer = None

def get_db_connection():
    """Подключение к базе данных"""
    return sqlite3.connect(DB_PATH)

def init_database():
    """Инициализация базы данных"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблица пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        language TEXT DEFAULT 'ru',
        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Таблица чатов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chats (
        chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_name TEXT UNIQUE,
        telegram_group_id TEXT,
        member_count INTEGER DEFAULT 0,
        is_active BOOLEAN DEFAULT TRUE,
        keywords TEXT DEFAULT '[]'
    )
    ''')

    # Таблица участия пользователей в чатах
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_chats (
        user_id INTEGER,
        chat_id INTEGER,
        join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (chat_id) REFERENCES chats (chat_id),
        PRIMARY KEY (user_id, chat_id)
    )
    ''')

    # Таблица пула интересов (сохраняем интересы пользователей для будущих чатов)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS interest_pool (
        interest_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        topic_name TEXT,
        query_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending',
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')

    # Таблица сообщений поддержки
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS support_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        user_message TEXT,
        admin_response TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'new',
        FOREIGN KEY (user_id) REFERENCES users (user_id)
    )
    ''')

    # Добавляем предопределенные темы
    for topic, group_id in GROUP_IDS.items():
        cursor.execute('''
        INSERT OR IGNORE INTO chats (chat_name, telegram_group_id, keywords) 
        VALUES (?, ?, ?)
        ''', (topic, group_id, json.dumps(DETAILED_TOPICS[topic]['keywords'])))
        
    conn.commit()
    conn.close()
    logger.info("✅ База данных инициализирована")

def preload_nlp_models():
    """Предзагрузка NLP моделей для ускорения работы (облегченная версия)"""
    global topic_vectors, vectorizer
    
    logger.info("🔄 Загрузка NLP моделей (облегченная версия)...")
    
    try:
        # Используем TF-IDF вместо тяжелых эмбеддингов
        vectorizer = TfidfVectorizer(
            stop_words=list(stop_words_ru) + list(stop_words_en),
            max_features=1000,
            ngram_range=(1, 2)
        )
        
        # Подготавливаем тексты для обучения
        topic_texts = []
        topic_names = []
        
        for topic, data in DETAILED_TOPICS.items():
            # Объединяем название, ключевые слова и описание
            keywords = " ".join(data['keywords'])
            description = data['description']
            full_text = f"{topic} {keywords} {description}"
            
            topic_texts.append(full_text)
            topic_names.append(topic)
        
        # Обучаем TF-IDF
        logger.info("🔄 Обучение TF-IDF векторизатора...")
        topic_vectors = vectorizer.fit_transform(topic_texts)
        
        logger.info(f"✅ TF-IDF модель обучена: {len(topic_names)} тем, {topic_vectors.shape[1]} признаков")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки NLP моделей: {e}")
        logger.info("⚠️ Работа в режиме базового поиска")
        vectorizer = None
        topic_vectors = None

def preprocess_text(text, language='ru'):
    """Предобработка текста для анализа"""
    # Определяем язык, если не указан
    if language == 'auto':
        try:
            language = detect(text)
        except:
            language = 'ru'
    
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Удаляем специальные символы и числа
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Токенизация
    tokens = word_tokenize(text)
    
    # Удаляем стоп-слова и выполняем стемминг
    if language.startswith('ru'):
        tokens = [stemmer_ru.stem(token) for token in tokens if token not in stop_words_ru and len(token) > 2]
    else:
        tokens = [stemmer_en.stem(token) for token in tokens if token not in stop_words_en and len(token) > 2]
    
    return " ".join(tokens), language

def find_best_matching_chat(user_query):
    """Интеллектуальный поиск наиболее подходящего чата"""
    try:
        logger.info(f"🔍 Поиск чата для запроса: '{user_query}'")
        
        # Определяем язык запроса
        detected_lang = detect(user_query) if len(user_query) > 3 else 'ru'
        
        # Предобработка запроса
        processed_query, query_lang = preprocess_text(user_query, detected_lang)
        
        # Шаг 1: Проверяем на точное совпадение с названиями чатов
        for chat_name, group_id in GROUP_IDS.items():
            if (user_query.lower() in chat_name.lower() or 
                chat_name.lower() in user_query.lower()):
                logger.info(f"✅ Найдено точное совпадение: {chat_name}")
                return chat_name, 1.0, "точное совпадение"
        
        # Шаг 2: Поиск по ключевым словам
        best_match = None
        best_score = 0.0
        
        query_words = set(processed_query.split())
        
        for topic, data in DETAILED_TOPICS.items():
            topic_keywords_set = set(word.lower() for word in data['keywords'])
            intersection = query_words.intersection(topic_keywords_set)
            
            if intersection:
                score = len(intersection) / len(topic_keywords_set)
                if score > best_score:
                    best_score = score
                    best_match = topic
        
        if best_match and best_score >= 0.3:
            logger.info(f"✅ Найдено совпадение по ключевым словам: {best_match} (score: {best_score:.2f})")
            return best_match, best_score, "совпадение по ключевым словам"
        
        # Шаг 3: TF-IDF поиск
        if vectorizer is not None and topic_vectors is not None:
            # Преобразуем запрос в TF-IDF вектор
            query_vector = vectorizer.transform([processed_query])
            
            # Вычисляем косинусное сходство
            similarities = cosine_similarity(query_vector, topic_vectors)
            
            # Находим максимальное сходство
            max_similarity_idx = similarities.argmax()
            max_similarity = similarities[0, max_similarity_idx]
            
            if max_similarity > 0.15:  # Повышен порог для лучшей точности
                best_match = list(DETAILED_TOPICS.keys())[max_similarity_idx]
                logger.info(f"✅ Найдено TF-IDF совпадение: {best_match} (score: {max_similarity:.2f})")
                return best_match, float(max_similarity), "похожая тематика"
        
        # Шаг 4: Поиск по ключевым терминам (пониженный порог)
        main_themes = {
            "путешествие": "Путешествие и туризм",
            "экономика": "Экономика и Бизнес",
            "здоровье": "Здоровье и медицина",
            "программирование": "Программирование",
            "искусство": "Искусство и музыка",
            "кулинария": "Кулинария и рецепты",
            "спорт": "Спорт",
            "наука": "Наука и литература",
            "образование": "Образование и Саморазвитие",
            "финансы": "Экономика и Бизнес",
            "деньги": "Экономика и Бизнес",
            "бизнес": "Экономика и Бизнес",
            "книги": "Наука и литература",
            "фитнес": "Спорт",
            "музыка": "Искусство и музыка",
            "живопись": "Искусство и музыка",
            "готовка": "Кулинария и рецепты",
            "туризм": "Путешествие и туризм",
            "развитие": "Образование и Саморазвитие",
            "психология": "Образование и Саморазвитие"
        }
        
        for keyword, topic in main_themes.items():
            if keyword in user_query.lower():
                logger.info(f"✅ Найден ключевой термин '{keyword}', предлагаю тему: {topic}")
                return topic, 0.4, f"ключевой термин: {keyword}"
        
        # Если ничего не нашли
        logger.info("❌ Подходящий чат не найден")
        return None, 0.0, "не найдено"
        
    except Exception as e:
        logger.error(f"❌ Ошибка при поиске чата: {e}")
        return None, 0.0, "ошибка поиска"

def get_invite_link_simple(group_id, bot_token):
    """Получение инвайт-ссылки через API запрос"""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/createChatInviteLink"
        params = {
            'chat_id': group_id,
            'member_limit': 1,
            'name': f'Инвайт от бота {datetime.now().strftime("%Y%m%d")}'
        }
        
        response = requests.post(url, data=params, timeout=15)
        data = response.json()
        
        if data.get('ok'):
            return data['result']['invite_link']
        else:
            error_msg = data.get('description', 'Неизвестная ошибка')
            logger.error(f"❌ Ошибка Telegram API: {error_msg}")
            return f"❌ Ошибка Telegram: {error_msg}"
            
    except Exception as e:
        logger.error(f"❌ Ошибка при получении ссылки: {e}")
        return f"❌ Сетевая ошибка: {str(e)}"

def get_main_menu_keyboard():
    """Получение клавиатуры главного меню"""
    keyboard = [
        [KeyboardButton("🔍 Найти группу по интересам")],
        [KeyboardButton("📋 Мои группы"), KeyboardButton("👤 Профиль")],
        [KeyboardButton("🎯 Популярные темы"), KeyboardButton("❓ Помощь")],
        [KeyboardButton("🆘 Поддержка")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

def get_popular_topics_keyboard():
    """Получение клавиатуры популярных тем с эмодзи"""
    keyboard = []
    
    # Группируем темы по 2 в строке
    for i in range(0, len(DETAILED_TOPICS), 2):
        row = []
        for j in range(i, min(i+2, len(DETAILED_TOPICS))):
            topic = list(DETAILED_TOPICS.keys())[j]
            emoji = DETAILED_TOPICS[topic]['emoji']
            row.append(KeyboardButton(f"{emoji} {topic}"))
        keyboard.append(row)
    
    keyboard.append([KeyboardButton("🔙 Назад"), KeyboardButton("❌ Отказаться")])
    
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Приветствие с умным меню"""
    user = update.message.from_user
    
    # Определяем язык пользователя
    try:
        user_lang = update.message.from_user.language_code
    except:
        user_lang = 'ru'
    
    # Сохраняем пользователя в БД
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT OR REPLACE INTO users (user_id, username, first_name, language, last_active, registration_date)
    VALUES (?, ?, ?, ?, datetime('now'), COALESCE((SELECT registration_date FROM users WHERE user_id = ?), datetime('now')))
    ''', (user.id, user.username, user.first_name, user_lang[:2], user.id))
    conn.commit()
    conn.close()
    
    welcome_text = f"""
🤖 **Привет, {user.first_name}!**
    
🌟 **Я - ваш личный гид по миру единомышленников!**

Здесь люди с общими интересами:
✅ Создают совместные проекты
✅ Обсуждают идеи и находят решения  
✅ Развиваются вместе и поддерживают друг друга
✅ Делятся знаниями и опытом

🎯 **Что вас интересует сегодня?** Выберите действие из меню ниже 👇
"""
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора в главном меню"""
    user_input = update.message.text.strip()
    user_id = update.message.from_user.id
    
    # Обработка естественных запросов
    if user_input.lower() in ["привет", "здравствуй", "hello", "hi", "привет!", "здравствуй!"]:
        return await start_command(update, context)
    
    if user_input.lower() in ["пока", "до свидания", "пока!", "до свидания!"]:
        goodbye_text = """
👋 **До свидания!** 

💡 **Не забывайте:** Вы всегда можете вернуться, нажав /start в любое время.

🌟 **Ждем вас снова!**
"""
        await update.message.reply_text(goodbye_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
        return MAIN_MENU
    
    # Обработка команд меню
    if user_input == "🔍 Найти группу по интересам":
        await update.message.reply_text(
            "🎯 **Что вас интересует?**\n\n"
            "Напишите тему, например:\n"
            "• 'путешествия по Азии'\n"
            "• 'программирование на Python'\n"
            "• 'здоровое питание'\n"
            "• 'фотография и дизайн'\n\n"
            "💡 **Или просто напишите ключевое слово:** 'путешествия', 'спорт', 'книги'",
            parse_mode='Markdown',
            reply_markup=ReplyKeyboardRemove()
        )
        return ASK_TOPIC
    
    elif user_input == "📋 Мои группы":
        return await groups_command(update, context)
    
    elif user_input == "👤 Профиль":
        return await profile_command(update, context)
    
    elif user_input == "🎯 Популярные темы":
        return await show_popular_topics(update, context)
    
    elif user_input == "❓ Помощь":
        return await help_command(update, context)
    
    elif user_input == "🆘 Поддержка":
        return await support_command(update, context)
    
    else:
        # Проверяем, не является ли это командой
        if user_input.startswith('/'):
            await update.message.reply_text(
                "❓ **Неизвестная команда.** Используйте меню для выбора действия.",
                parse_mode='Markdown',
                reply_markup=get_main_menu_keyboard()
            )
            return MAIN_MENU
        
        # Если пользователь просто написал что-то в главном меню - предлагаем поиск
        await update.message.reply_text(
            "🔍 **Хотите найти группу по вашему запросу?**\n\n"
            "Нажмите кнопку ниже для поиска групп по интересам!",
            parse_mode='Markdown',
            reply_markup=get_main_menu_keyboard()
        )
        return MAIN_MENU

async def handle_ask_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка ввода темы с интеллектуальным поиском"""
    user_topic = update.message.text.strip()
    
    await update.message.reply_text(
        "🧠 **Анализирую ваш запрос...**\n\nЭто может занять 10-15 секунд. Я ищу самые релевантные группы для вас.",
        parse_mode='Markdown'
    )
    
    chat_name, score, reason = find_best_matching_chat(user_topic)
    
    # Сохраняем запрос пользователя в базу для учета интересов
    if user_topic:
        user_id = update.message.from_user.id
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO interest_pool (user_id, topic_name, query_text)
        VALUES (?, ?, ?)
        ''', (user_id, user_topic[:100], user_topic[:500]))
        conn.commit()
        conn.close()
    
    if chat_name:
        # Убираем технические детали для пользователя
        if reason == "точное совпадение":
            reason_text = "идеально подходит под ваш запрос"
        elif reason == "совпадение по ключевым словам":
            reason_text = "совпадает с вашими интересами"
        elif reason == "похожая тематика":
            reason_text = "похожа на ваш запрос"
        elif "ключевой термин" in reason:
            reason_text = "содержит ключевые слова из вашего запроса"
        else:
            reason_text = "может быть интересна вам"
        
        await update.message.reply_text(
            f"🎯 **Отлично! Я нашел идеальную группу для вас!**\n\n"
            f"**Тема:** {chat_name}\n"
            f"**Почему эта группа:** {reason_text}\n\n"
            f"**Описание:** {DETAILED_TOPICS[chat_name]['description']}\n\n"
            f"Хотите присоединиться к группе «{chat_name}»?",
            parse_mode='Markdown',
            reply_markup=ReplyKeyboardMarkup([
                [KeyboardButton("✅ Присоединиться"), KeyboardButton("❌ Отказаться")]
            ], resize_keyboard=True)
        )
        context.user_data['selected_chat'] = chat_name
        context.user_data['user_topic'] = user_topic
        return JOIN_CHAT
    else:
        # Сохраняем интерес пользователя для будущих чатов
        user_id = update.message.from_user.id
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO interest_pool (user_id, topic_name, query_text)
        VALUES (?, ?, ?)
        ''', (user_id, "новый интерес", user_topic[:500]))
        conn.commit()
        conn.close()
        
        no_match_text = f"""
🔍 **К сожалению, я не нашел подходящей группы по запросу «{user_topic}».**

💡 **Мы учтем ваш интерес!** 
Мы сохранили ваш запрос и, возможно, скоро откроем такой чат.

🎯 **А пока, можете выбрать из популярных тем:**
"""
        await update.message.reply_text(
            no_match_text,
            parse_mode='Markdown',
            reply_markup=get_popular_topics_keyboard()
        )
        return CHOOSE_TOPIC

async def handle_join_decision(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка решения о присоединении"""
    user_decision = update.message.text.strip()
    user_id = update.message.from_user.id
    
    if user_decision == "❌ Отказаться":
        await update.message.reply_text(
            "👋 **Хорошо, вы отказались от присоединения.**\n\n"
            "💡 **Это абсолютно нормально!** Вы можете найти другую группу или вернуться позже.\n\n"
            "🎯 **Что дальше?**",
            parse_mode='Markdown',
            reply_markup=get_main_menu_keyboard()
        )
        return MAIN_MENU
    
    if user_decision == "🏠 В меню":
        await update.message.reply_text(
            "🏠 **Вы вернулись в главное меню**\n\nВыберите действие:",
            parse_mode='Markdown',
            reply_markup=get_main_menu_keyboard()
        )
        return MAIN_MENU
    
    if user_decision == "✅ Присоединиться":
        chat_name = context.user_data.get('selected_chat')
        if not chat_name:
            await update.message.reply_text(
                "❌ **Ошибка: чат не выбран.** Начните поиск заново с главного меню.",
                parse_mode='Markdown',
                reply_markup=get_main_menu_keyboard()
            )
            return MAIN_MENU
        
        group_id = GROUP_IDS.get(chat_name)
        if not group_id:
            await update.message.reply_text(
                f"❌ **Ошибка: группа «{chat_name}» не найдена в базе.** Пожалуйста, сообщите об этой ошибке в поддержку.",
                parse_mode='Markdown',
                reply_markup=get_main_menu_keyboard()
            )
            return MAIN_MENU
        
        # Получаем инвайт-ссылку
        invite_link = get_invite_link_simple(group_id, BOT_TOKEN)
        
        if invite_link.startswith("https://t.me/"):
            # Добавляем пользователя в чат
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Получаем chat_id из базы
            cursor.execute('SELECT chat_id FROM chats WHERE chat_name = ?', (chat_name,))
            result = cursor.fetchone()
            chat_db_id = result[0] if result else None
            
            if chat_db_id:
                cursor.execute('''
                INSERT OR IGNORE INTO user_chats (user_id, chat_id) 
                VALUES (?, ?)
                ''', (user_id, chat_db_id))
                
                # Увеличиваем счетчик участников
                cursor.execute('''
                UPDATE chats SET member_count = member_count + 1 
                WHERE chat_id = ?
                ''', (chat_db_id,))
                
                conn.commit()
                success = True
            else:
                success = False
            
            conn.close()
            
            if success:
                success_text = f"""
🎉 **Отлично! Вы успешно присоединились к группе «{chat_name}»!**

🔗 **Ваша персональная ссылка:** {invite_link}

🌟 **Что дальше:**
• Нажмите на ссылку, чтобы войти в чат
• Представьтесь участникам
• Начните обсуждение или задайте вопрос
• Найдите единомышленников для проектов

💡 **Совет:** Активное участие поможет вам быстрее найти друзей и партнеров!

🔄 **Хотите найти еще одну группу по другим интересам?** Нажмите "🔍 Найти группу по интересам" в меню!
"""
                await update.message.reply_text(success_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
                return MAIN_MENU
            else:
                await update.message.reply_text(
                    "❌ **Ошибка при добавлении в базу данных.** Попробуйте позже.",
                    parse_mode='Markdown',
                    reply_markup=get_main_menu_keyboard()
                )
                return MAIN_MENU
        else:
            error_text = f"""
⚠️ **Не удалось получить ссылку для группы «{chat_name}»**

❌ **Причина:** {invite_link}

🔧 **Что проверить:**
1. ID группы: `{group_id}`
2. Бот добавлен в группу как администратор
3. У бота есть права: `invite users`

🔄 **Выберите другую тему для поиска:**
"""
            await update.message.reply_text(error_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
            return MAIN_MENU
    
    if user_decision == "🔄 Другие варианты":
        return await show_popular_topics(update, context)
    
    # Если неизвестная команда
    await update.message.reply_text(
        "❓ **Неизвестная команда.** Пожалуйста, используйте кнопки для выбора действия.",
        parse_mode='Markdown',
        reply_markup=get_main_menu_keyboard()
    )
    return MAIN_MENU

async def show_popular_topics(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показ популярных тем с эмодзи"""
    response_text = "🎯 **Выберите интересующую вас тему из популярных:**"
    
    await update.message.reply_text(response_text, reply_markup=get_popular_topics_keyboard(), parse_mode='Markdown')
    return CHOOSE_TOPIC

async def handle_popular_topic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка выбора популярной темы"""
    user_input = update.message.text.strip()
    
    if user_input == "❌ Отказаться":
        goodbye_text = """
👋 **До свидания!** 

💡 **Не забывайте:** Вы всегда можете вернуться, нажав /start в любое время.

🌟 **Ждем вас снова!**
"""
        await update.message.reply_text(goodbye_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
        return MAIN_MENU
    
    if user_input == "🔙 Назад":
        await update.message.reply_text(
            "🏠 **Вы вернулись в главное меню**\n\nВыберите действие:",
            parse_mode='Markdown',
            reply_markup=get_main_menu_keyboard()
        )
        return MAIN_MENU
    
    # Извлекаем название темы из кнопки с эмодзи
    topic_name = user_input.split(' ', 1)[-1] if ' ' in user_input else user_input
    
    if topic_name in GROUP_IDS:
        chat_name = topic_name
        await update.message.reply_text(
            f"🎯 **Отличный выбор!**\n\n"
            f"**Тема:** {chat_name}\n"
            f"**Описание:** {DETAILED_TOPICS[chat_name]['description']}\n\n"
            f"👥 **Участники уже обсуждают:**\n"
            f"• {', '.join(DETAILED_TOPICS[chat_name]['keywords'][:3])}\n\n"
            f"Хотите присоединиться к группе «{chat_name}»?",
            parse_mode='Markdown',
            reply_markup=ReplyKeyboardMarkup([
                [KeyboardButton("✅ Присоединиться"), KeyboardButton("❌ Отказаться")],
                [KeyboardButton("🔄 Другие темы"), KeyboardButton("🏠 В меню")]
            ], resize_keyboard=True)
        )
        context.user_data['selected_chat'] = chat_name
        return JOIN_CHAT
    else:
        await update.message.reply_text(
            f"⚠️ **Группа «{topic_name}» временно недоступна.** Выберите другую тему:",
            parse_mode='Markdown',
            reply_markup=get_popular_topics_keyboard()
        )
        return CHOOSE_TOPIC

async def groups_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показ групп пользователя"""
    user_id = update.message.from_user.id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT c.chat_name 
    FROM chats c
    JOIN user_chats uc ON c.chat_id = uc.chat_id
    WHERE uc.user_id = ? AND c.is_active = 1
    ''', (user_id,))
    user_chats = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    if not user_chats:
        no_groups_text = """
❌ **Вы пока не состоите ни в одной группе**

🎯 **Как найти свою первую группу:**
1. Нажмите "🔍 Найти группу по интересам" в главном меню
2. Напишите, чем вы увлекаетесь
3. Выберите подходящий чат из предложенных
"""
        await update.message.reply_text(no_groups_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
        return MAIN_MENU
    
    groups_text = """
📋 **Ваши группы**

🌟 **Вы состоите в следующих чатах:**
"""
    
    for i, chat_name in enumerate(user_chats, 1):
        groups_text += f"{i}. {chat_name}\n"
    
    groups_text += f"\n💬 **Всего групп:** {len(user_chats)}"
    
    await update.message.reply_text(groups_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показ профиля пользователя"""
    user = update.message.from_user
    user_id = user.id
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Получаем данные пользователя
    cursor.execute('''
    SELECT username, first_name, language, last_active, registration_date,
           (SELECT COUNT(*) FROM user_chats WHERE user_id = ?) as group_count,
           (SELECT COUNT(*) FROM interest_pool WHERE user_id = ?) as interests_count
    FROM users 
    WHERE user_id = ?
    ''', (user_id, user_id, user_id))
    
    user_data = cursor.fetchone()
    conn.close()
    
    if user_data:
        username, first_name, language, last_active, registration_date, group_count, interests_count = user_data
        
        # Форматируем даты
        try:
            last_active_formatted = datetime.strptime(last_active, '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y %H:%M')
            registration_formatted = datetime.strptime(registration_date, '%Y-%m-%d %H:%M:%S').strftime('%d.%m.%Y')
        except:
            last_active_formatted = last_active
            registration_formatted = registration_date
        
        profile_text = f"""
👤 **Ваш профиль**

📝 **Основная информация:**
• ID: `{user_id}`
• Имя: {first_name}
• Username: @{username if username else 'не указан'}
• Язык: {language}
• Дата регистрации: {registration_formatted}
• Последняя активность: {last_active_formatted}

📊 **Статистика:**
• Активных групп: {group_count}
• Найдено интересов: {interests_count}
• Начато поисков: {interests_count}

🏆 **Достижения:**
{'• Знакомство с ботом ✅' if group_count >= 0 else ''}
{'• Первая группа ✅' if group_count >= 1 else '• Первая группа ⏳'}
{'• Активный участник ✅' if group_count >= 3 else '• Активный участник ⏳'}
{'• Лидер сообщества ✅' if group_count >= 5 else '• Лидер сообщества ⏳'}

⚙️ **Настройки:**
• Уведомления: включены
• Язык интерфейса: русский
• Темная тема: по умолчанию

💡 **Совет:** Чем больше групп вы попробуете, тем точнее бот сможет рекомендовать вам интересные темы!
"""
    else:
        profile_text = """
👤 **Профиль не найден**

Пожалуйста, начните с команды /start
"""
    
    await update.message.reply_text(profile_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показ справки"""
    help_text = """
📖 **Справка по боту**

🎯 **Как это работает:**
1. Вы указываете тему, которая вас интересует
2. Бот **умно ищет** подходящие чаты по ключевым словам
3. Если находит - предлагает присоединиться
4. Если нет - уточняет тему или предлагает похожие варианты
5. Популярные темы всегда доступны в меню

🧠 **Умный поиск:**
• Бот понимает **синонимы** (деньги → экономика)
• Анализирует **контекст** (заработок → бизнес)
• Ищет **похожие темы** при неточных совпадениях
• Предлагает **релевантные варианты** даже если точного совпадения нет

💡 **Важно:**
• Бот добавляет вас только в тематические чаты
• Ссылки для приглашения одноразовые
• Вы всегда можете отказаться от присоединения

🆘 **Поддержка:**
Напишите /support для обращения к администратору
"""
    await update.message.reply_text(help_text, parse_mode='Markdown', reply_markup=get_main_menu_keyboard())
    return MAIN_MENU

async def support_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка команды поддержки"""
    support_text = """
🆘 **Поддержка**

Напишите ваш вопрос или проблему, и я передам сообщение администратору.

⚠️ **Важно:** Это не техническая поддержка Telegram, а поддержка именно этого бота.

✏️ **Введите ваше сообщение ниже:**
"""
    await update.message.reply_text(support_text, parse_mode='Markdown', reply_markup=ReplyKeyboardMarkup([
        [KeyboardButton("🏠 В меню"), KeyboardButton("❌ Отмена")]
    ], resize_keyboard=True))
    return SUPPORT

async def handle_support_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработка сообщения для поддержки"""
    user_message = update.message.text.strip()
    user_id = update.message.from_user.id
    username = update.message.from_user.username
    first_name = update.message.from_user.first_name
    
    if user_message == "🏠 В меню":
        await update.message.reply_text(
            "🏠 **Вы вернулись в главное меню**\n\nВыберите действие:",
            parse_mode='Markdown',
            reply_markup=get_main_menu_keyboard()
        )
        return MAIN_MENU
    
    elif user_message == "❌ Отмена":
        await update.message.reply_text(
            "❌ **Отправка в поддержку отменена.**\n\nВыберите действие:",
            parse_mode='Markdown',
            reply_markup=get_main_menu_keyboard()
        )
        return MAIN_MENU
    
    else:
        # Сохраняем сообщение в базу данных
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO support_messages (user_id, user_message, status)
        VALUES (?, ?, ?)
        ''', (user_id, user_message, 'new'))
        conn.commit()
        conn.close()
        
        # Пересылаем сообщение админу
        try:
            admin_message = f"""
🆘 **НОВОЕ ОБРАЩЕНИЕ В ПОДДЕРЖКУ**

👤 **Пользователь:**
ID: `{user_id}`
Имя: {first_name}
Username: @{username if username else 'не указан'}

📝 **Сообщение:**
{user_message}

⏰ **Время:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=admin_message,
                parse_mode='Markdown'
            )
            
            await update.message.reply_text(
                "✅ **Ваше сообщение отправлено администратору!**\n\n"
                "Мы ответим вам в ближайшее время.\n\n"
                "Спасибо за обращение!",
                parse_mode='Markdown',
                reply_markup=get_main_menu_keyboard()
            )
            return MAIN_MENU
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сообщения админу: {e}")
            # Сообщение уже сохранено в БД, админ сможет прочитать позже
            await update.message.reply_text(
                "✅ **Ваше сообщение сохранено!**\n\n"
                "Администратор получит его, как только будет онлайн, и ответит вам.\n\n"
                "Спасибо за обращение!",
                parse_mode='Markdown',
                reply_markup=get_main_menu_keyboard()
            )
            return MAIN_MENU

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка ошибок"""
    logger.error(f"Ошибка при обработке обновления {update}: {context.error}")
    
    try:
        if update and update.message:
            await update.message.reply_text(
                "❌ **Произошла ошибка при обработке вашего запроса.**\n\n"
                "Попробуйте еще раз или используйте команду /start",
                parse_mode='Markdown'
            )
    except:
        pass

def cleanup():
    """Очистка при завершении работы"""
    logger.info("🧹 Очистка ресурсов...")
    # Закрываем соединения с базой данных и т.д.

def main():
    """Основная функция запуска бота"""
    logger.info("🚀 Запуск Telegram бота с интеллектуальным поиском...")
    
    # Регистрируем очистку при завершении
    atexit.register(cleanup)
    
    # Проверяем наличие токена
    if not BOT_TOKEN:
        logger.critical("❌ BOT_TOKEN не найден!")
        logger.critical("Добавьте BOT_TOKEN в переменные окружения Railway")
        sys.exit(1)
    
    # Инициализация базы данных
    init_database()
    
    # Загрузка NLP моделей
    preload_nlp_models()
    
    try:
        # Создаем приложение
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Добавляем обработчик ошибок
        application.add_error_handler(error_handler)
        
        # Добавляем обработчики
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', start_command)],
            states={
                MAIN_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_main_menu)],
                ASK_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ask_topic)],
                CHOOSE_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_popular_topic)],
                JOIN_CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_join_decision)],
                SUPPORT: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_support_message)],
            },
            fallbacks=[
                CommandHandler('start', start_command),
                CommandHandler('help', help_command),
                CommandHandler('profile', profile_command),
                CommandHandler('support', support_command),
                CommandHandler('groups', groups_command),
                MessageHandler(filters.TEXT, handle_main_menu)
            ],
            allow_reentry=True
        )
        
        application.add_handler(conv_handler)
        application.add_handler(CommandHandler('help', help_command))
        application.add_handler(CommandHandler('groups', groups_command))
        application.add_handler(CommandHandler('support', support_command))
        application.add_handler(CommandHandler('profile', profile_command))
        
        logger.info("✅ Бот успешно инициализирован")
        logger.info("⚡ Бот запущен и готов к приему сообщений!")
        
        # Запускаем в режиме polling
        application.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
        
    except Exception as e:
        logger.critical(f"🔥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        logger.critical("Проверьте правильность BOT_TOKEN в переменных окружения Railway")
        raise

if __name__ == "__main__":
    main()
