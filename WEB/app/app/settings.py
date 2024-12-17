import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("web.env")  # 환경 변수 로드

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("USE_DEBUG", "FALSE").upper() == "TRUE"

ALLOWED_HOSTS = [
    "15.165.31.54",  # 서버 IP 주소
    "mane.my",  # 메인 도메인
    ".mane.my",  # 서브도메인 허용
]

# HTTPS 설정
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000  # 1년 동안 HSTS 유지
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

CSRF_TRUSTED_ORIGINS = [
    "https://15.165.31.54",  # 서버 IP
    "https://mane.my",  # 메인 도메인
    "https://*.mane.my",  # 서브도메인
]

# Application definition
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "user",
    "report",
    "storages",  # S3 스토리지 사용
    "boto3",
    "rest_framework",
    "channels",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "app.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "app.wsgi.application"

# # Database
# if DEBUG:
#     DATABASES = {
#         "default": {
#             "ENGINE": "django.db.backends.postgresql",
#             "NAME": os.getenv("DB_DEBUG_NAME"),
#             "USER": os.getenv("DB_DEBUG_USER"),
#             "PASSWORD": os.getenv("DB_DEBUG_PASSWORD"),
#             "HOST": os.getenv("DB_DEBUG_HOST"),
#             "PORT": os.getenv("DB_DEBUG_PORT"),
#         }
#     }
# else:
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME"),
        "USER": os.getenv("DB_USER"),
        "PASSWORD": os.getenv("DB_PASSWORD"),
        "HOST": os.getenv("DB_HOST"),
        "PORT": os.getenv("DB_PORT"),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Internationalization
LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

# Static and Media Files
USE_S3 = os.getenv("USE_S3", "FALSE").upper() == "TRUE"

if USE_S3:
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
    AWS_S3_CUSTOM_DOMAIN = f"{AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com"
    AWS_QUERYSTRING_AUTH = False  # URL 서명 제거
    AWS_S3_OBJECT_PARAMETERS = {"CacheControl": "max-age=86400"}  # 캐시 설정

    STATIC_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/static/"
    STATICFILES_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
else:
    STATIC_URL = "/staticfiles/"
    STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "/mediafiles/"
MEDIA_ROOT = os.path.join(BASE_DIR, "mediafiles")

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ASGI and Channels 설정
ASGI_APPLICATION = "app.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}
