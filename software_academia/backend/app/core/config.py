import os

class Settings:
    PROJECT_NAME: str = "Software Academia"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/software_academia")

settings = Settings()
