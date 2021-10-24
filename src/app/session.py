import random
from datetime import date

from src.app.cache import Cache
from src.app.settings import Settings


class Session:
    def __init__(self, settings: Settings):
        cache: Cache = settings.get('cache')
        self.id = cache.read('session_id', f'session_{random.randint(0, 99999)}')
        self.created_at = cache.read('created_at', f'{date.today()}')
        self.last_run_at = cache.read('last_run_at', absent_ok=True)
        self.cache = cache
        cache.save()

    def write(self, key, obj):
        self.cache.write(key, obj)

    def read(self, key):
        return self.cache.read(key, absent_ok=True)
