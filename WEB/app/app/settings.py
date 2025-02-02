import os
from pathlib import Path
from dotenv import load_dotenv
from distutils.util import strtobool

load_dotenv()  # 환경 변수 로드

# Build paths inside the project
BASE_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")

USE_DEBUG = bool(strtobool(os.getenv("USE_DEBUG", "False")))
DEBUG = USE_DEBUG

print("DEBUG : ", DEBUG)
ALLOWED_HOSTS = ["*"]

X_FRAME_OPTIONS = "SAMEORIGIN"
SECURE_SSL_REDIRECT = False
SECURE_PROXY_SSL_HEADER = None
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False

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
    "storages",
    "boto3",
    "rest_framework",
    "channels",
    "corsheaders",
    "neo4j",
    "drf_yasg",
    "quiz",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_CREDENTIALS = True

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

# Database
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
USE_S3 = bool(strtobool(os.getenv("USE_S3", "True")))
print("USE S3 : ", USE_S3)

if USE_S3:
    # AWS S3 설정
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_STORAGE_BUCKET_NAME = os.getenv("AWS_STORAGE_BUCKET_NAME")
    AWS_REGION = os.getenv("AWS_S3_REGION_NAME", "ap-northeast-2")  # 기본값은 서울 리전
    AWS_S3_CUSTOM_DOMAIN = f"{AWS_STORAGE_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com"
    AWS_QUERYSTRING_AUTH = False  # URL 서명 제거
    AWS_S3_OBJECT_PARAMETERS = {"CacheControl": "max-age=86400"}  # 캐시 설정

    # STORAGES 설정
    STORAGES = {
        "default": {
            "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",  # Media 파일 저장소
        },
        "staticfiles": {
            "BACKEND": "storages.backends.s3boto3.S3Boto3Storage",  # Static 파일 저장소
        },
    }

    # Static 및 Media 파일 URL
    STATIC_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/static/"
    MEDIA_URL = f"https://{AWS_S3_CUSTOM_DOMAIN}/media/"
else:
    # 로컬 개발 환경 설정
    STORAGES = {
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",  # Media 파일 저장소
            "LOCATION": os.path.join(BASE_DIR, "mediafiles"),
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",  # Static 파일 저장소
        },
    }

# Static 및 Media 파일 URL
STATIC_URL = "/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")
MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "mediafiles")

# STATICFILES_DIRS는 항상 사용 (소스 디렉토리)
STATICFILES_DIRS = [
    BASE_DIR / "static",  # 프로젝트 내 정적 파일 경로
]


# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ASGI and Channels 설정
ASGI_APPLICATION = "app.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}

# LOGGING 설정 추가
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
        "simple": {
            "format": "{levelname} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "file": {
            "level": "DEBUG",  # 로그 레벨
            "class": "logging.FileHandler",
            "filename": os.path.join(BASE_DIR, "logs/report.log"),  # report 앱의 로그 파일
            "formatter": "verbose",  # 사용할 포매터 지정
        },
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "simple",
        },
    },
    "loggers": {
        "django": {
            "handlers": ["console"],  # Django 기본 로그는 콘솔에 출력
            "level": "INFO",  # INFO 이상만 기록
            "propagate": True,
        },
        "report": {  # report 앱에 대한 로거
            "handlers": ["file"],  # report 앱의 로그는 파일에 저장
            "level": "DEBUG",  # DEBUG 이상 로그 기록
            "propagate": False,  # 상위 로거로 전달하지 않음
        },
    },
}

SWAGGER_SETTINGS = {
    "SECURITY_DEFINITIONS": {
        "Bearer": {"type": "apiKey", "name": "Authorization", "in": "header"}
    }
}
