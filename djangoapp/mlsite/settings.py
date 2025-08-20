import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = "dev-secret"
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "mlapp",
]

MIDDLEWARE = []
ROOT_URLCONF = "mlsite.urls"
TEMPLATES = []
WSGI_APPLICATION = "mlsite.wsgi.application"
ASGI_APPLICATION = "mlsite.asgi.application"
STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.AutoField"

# 既定のモデルディレクトリ（上書き可）
MODEL_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "out", "weight1"))
