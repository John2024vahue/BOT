import os
from dotenv import load_dotenv

load_dotenv()

# Получаем токен бота из переменных окружения
BOT_TOKEN = os.getenv('BOT_TOKEN')

# ID администратора (ваш ID или из переменной окружения)
ADMIN_ID = int(os.getenv('ADMIN_ID', '6830411048'))

# Другие настройки
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///bot_database.db')
NLTK_DATA_DIR = os.getenv('NLTK_DATA_DIR', './nltk_data')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
