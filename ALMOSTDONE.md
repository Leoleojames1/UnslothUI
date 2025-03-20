# Complete UnslothUI GitHub Repository Setup

I'll guide you through creating a complete, production-ready GitHub repository structure for the UnslothUI project. Follow these steps to organize all the files and prepare them for GitHub.

## 1. Repository Structure

First, create the main folder structure:

```bash
# Create main directory
mkdir -p unslothui
cd unslothui

# Create subdirectories
mkdir -p .github/workflows
mkdir -p alembic/versions
mkdir -p api
mkdir -p core
mkdir -p integrations
mkdir -p models
mkdir -p services
mkdir -p tests/test_api
mkdir -p workers
mkdir -p storage/datasets storage/models storage/checkpoints
```

## 2. Initialize Git Repository

```bash
# Initialize git repository
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Database
*.db
*.sqlite3

# Logs
*.log
logs/

# Environment variables
.env

# Storage
storage/

# Alembic
alembic/versions/*.py
!alembic/versions/__init__.py

# Models and datasets
models/
datasets/
checkpoints/

# Test coverage
.coverage
htmlcov/
EOF

# Create .dockerignore
cat > .dockerignore << 'EOF'
.git
.github
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
*.log
.coverage
.gitlab-ci.yml
.gitignore
.dockerignore
Dockerfile
docker-compose.yml
tests/
*.db
storage/
.pytest_cache/
htmlcov/
EOF
```

## 3. Create Root Files

```bash
# Create README.md
cat > README.md << 'EOF'
# UnslothUI

UnslothUI is a web interface for [Unsloth](https://github.com/unslothai/unsloth), a library that enables faster and more memory-efficient fine-tuning of Large Language Models (LLMs).

## Features

- 🚀 **Efficient Fine-tuning**: Leverage Unsloth's optimizations for faster training with lower memory usage
- 🔄 **Real-time Monitoring**: Track training progress with WebSocket-based live metrics and logs
- 📊 **Dataset Management**: Upload, preprocess, and manage training datasets
- ⚙️ **Model Configuration**: Configure LoRA, quantization, and training parameters
- 🧠 **Hugging Face Integration**: Seamless model discovery, download, and upload
- 🔐 **User Authentication**: Secure multi-user support with JWT authentication

## Getting Started

### Prerequisites

- Python 3.10+ 
- CUDA-compatible GPU (for training)
- Redis server
- PostgreSQL (recommended for production) or SQLite

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unslothui.git
   cd unslothui
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (copy and modify .env.example):
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

4. Initialize the database:
   ```bash
   alembic upgrade head
   ```

5. Start the API server:
   ```bash
   python main.py
   ```

6. Start the worker process:
   ```bash
   python -m workers.training_worker
   ```

### Using Docker

For a production-ready setup, you can use Docker Compose:

```bash
docker-compose up -d
```

This will start:
- API server
- Worker process
- Redis
- PostgreSQL

## Documentation

API documentation is available at `http://localhost:8000/docs` when the server is running.

## Architecture

UnslothUI follows a client-server architecture with these key components:

- **Web UI**: React-based frontend (separate repository)
- **API Server**: FastAPI backend that handles requests
- **Training Worker**: Background process for running training jobs
- **Storage Layer**: Systems for storing models, datasets, and results
- **Authentication Service**: Handles user auth and permissions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for the core LLM optimization technology
- [Hugging Face](https://huggingface.co/) for model ecosystem integration
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
EOF

# Create LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 UnslothUI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi>=0.103.0
uvicorn>=0.23.2
pydantic>=2.3.0
pydantic-settings>=2.0.3
sqlalchemy>=2.0.20
alembic>=1.12.0
redis>=5.0.0
aiohttp>=3.8.5
python-jose>=3.3.0
passlib>=1.7.4
python-multipart>=0.0.6
huggingface-hub>=0.17.3
unsloth>=0.5.0
transformers>=4.32.0
accelerate>=0.22.0
datasets>=2.14.5
bitsandbytes>=0.41.1
peft>=0.5.0
websockets>=11.0.3
pandas>=2.0.0
pytest>=7.4.0
python-dotenv>=1.0.0
EOF

# Create requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
# Development dependencies
pytest>=7.4.0
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0
black>=23.7.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.1
httpx>=0.24.1
coverage>=7.3.1
sqlalchemy-utils>=0.41.1
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Application settings
DEBUG=False
APP_NAME=UnslothUI

# API settings
HOST=0.0.0.0
PORT=8000

# CORS settings
CORS_ORIGINS=http://localhost:3000,http://frontend.example.com

# Authentication settings
SECRET_KEY=change-this-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database settings
DATABASE_URL=postgresql://postgres:postgres@db:5432/unslothui
# For SQLite, use: DATABASE_URL=sqlite:///./unslothui.db

# Redis settings
REDIS_URL=redis://redis:6379/0

# Storage settings
STORAGE_TYPE=local
STORAGE_PATH=./storage
# For S3, uncomment and set:
# STORAGE_TYPE=s3
# S3_ENDPOINT=https://your-s3-endpoint
# S3_ACCESS_KEY=your-access-key
# S3_SECRET_KEY=your-secret-key
# S3_BUCKET_NAME=unslothui

# Hugging Face settings
HF_TOKEN=

# Training settings
MAX_CONCURRENT_JOBS=2
DEFAULT_TRAINING_DEVICE=cuda:0
EOF

# Create Docker files
cat > Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage /app/storage/datasets /app/storage/models /app/storage/checkpoints

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build: .
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - .:/app
      - shared_storage:/app/storage
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/unslothui
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    restart: unless-stopped

  worker:
    build: .
    command: python -m workers.training_worker
    volumes:
      - .:/app
      - shared_storage:/app/storage
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/unslothui
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  db:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=unslothui
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  shared_storage:
EOF

# Create main.py
cat > main.py << 'EOF'
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from api import auth, models, datasets, training, evaluation
from core.config import settings
from core.security import get_current_user

app = FastAPI(
    title="UnslothUI API",
    description="Backend API for UnslothUI, a web interface for Unsloth LLM fine-tuning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(
    models.router, 
    prefix="/api/v1", 
    tags=["Models"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    datasets.router, 
    prefix="/api/v1", 
    tags=["Datasets"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    training.router, 
    prefix="/api/v1", 
    tags=["Training"],
    dependencies=[Depends(get_current_user)]
)
app.include_router(
    evaluation.router, 
    prefix="/api/v1", 
    tags=["Evaluation"],
    dependencies=[Depends(get_current_user)]
)

@app.get("/", tags=["Health"])
async def root():
    return {"message": "UnslothUI API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
EOF

# Create GitHub Actions workflow file
mkdir -p .github/workflows
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis
        ports:
          - 6379:6379
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: unslothui_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Format check with black
      run: |
        black --check .
    
    - name: Test with pytest
      run: |
        pytest --cov=./ --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/unslothui_test
        REDIS_URL: redis://localhost:6379/0
        SECRET_KEY: test-secret-key
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false
EOF
```

## 4. Create Core Files

Now let's create the core modules:

```bash
# Create the __init__.py files
touch api/__init__.py core/__init__.py integrations/__init__.py models/__init__.py services/__init__.py
touch workers/__init__.py tests/__init__.py tests/test_api/__init__.py alembic/versions/__init__.py

# Create core/config.py
cat > core/config.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = os.getenv("APP_NAME", "UnslothUI")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    
    # Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./unslothui.db")
    
    # Redis settings
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Storage settings
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # "local", "s3", "minio"
    STORAGE_PATH: str = os.getenv("STORAGE_PATH", "./storage")
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "unslothui")
    
    # Hugging Face settings
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # Training settings
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    DEFAULT_TRAINING_DEVICE: str = os.getenv("DEFAULT_TRAINING_DEVICE", "cuda:0")
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

# Create core/database.py
cat > core/database.py << 'EOF'
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from core.config import settings

engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
EOF

# Create core/security.py
cat > core/security.py << 'EOF'
from datetime import datetime, timedelta
from typing import Optional, Union
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from core.config import settings
from core.database import get_db
from models.user import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 with password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "refresh": True})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")

async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = {"user_id": user_id}
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == token_data["user_id"]).first()
    if user is None:
        raise credentials_exception
    
    return user
EOF
```

## 5. Set Up Alembic

```bash
# Create alembic.ini
cat > alembic.ini << 'EOF'
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url = sqlite:///./unslothui.db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
EOF

# Create alembic/env.py
cat > alembic/env.py << 'EOF'
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import models to ensure they're picked up by Alembic
from models.user import User
from models.dataset import Dataset
from models.model import ModelConfig
from models.job import TrainingJob, Checkpoint

from core.config import settings
from core.database import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the SQLAlchemy URL with the one from settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
EOF
```

## 6. Create Database Models

Now let's create the database models:

```bash
# Create models/user.py
cat > models/user.py << 'EOF'
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    datasets = relationship("Dataset", back_populates="user")
    model_configs = relationship("ModelConfig", back_populates="user")
    training_jobs = relationship("TrainingJob", back_populates="user")
EOF

# Create models/dataset.py
cat > models/dataset.py << 'EOF'
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_path = Column(String)
    format = Column(String)  # json, csv, txt, etc.
    size_bytes = Column(Integer)
    row_count = Column(Integer)
    metadata = Column(JSON, nullable=True)
    is_preprocessed = Column(Boolean, default=False)
    preprocessing_config = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="datasets")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
EOF

# Create models/model.py
cat > models/model.py << 'EOF'
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base

class ModelConfig(Base):
    __tablename__ = "model_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    base_model_id = Column(String)  # Hugging Face model ID
    description = Column(Text, nullable=True)
    lora_config = Column(JSON)  # LoRA configurations
    training_config = Column(JSON)  # Training hyperparameters
    quantization_config = Column(JSON)  # Quantization settings
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="model_configs")
    training_jobs = relationship("TrainingJob", back_populates="model_config")
EOF

# Create models/job.py
cat > models/job.py << 'EOF'
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from core.database import Base

class JobStatus(enum.Enum):
    SUBMITTED = "submitted"
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_config_id = Column(Integer, ForeignKey("model_configs.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    status = Column(Enum(JobStatus), default=JobStatus.SUBMITTED)
    job_queue_id = Column(String, nullable=True)  # ID from job queue system
    process_id = Column(String, nullable=True)  # Process ID for running job
    device = Column(String)  # GPU device ID or CPU
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer)
    metrics = Column(JSON, nullable=True)  # Latest training metrics
    output_dir = Column(String, nullable=True)  # Directory for output files
    checkpoint_dir = Column(String, nullable=True)  # Directory for checkpoints
    error_message = Column(Text, nullable=True)  # Error message if failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="training_jobs")
    model_config = relationship("ModelConfig", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    checkpoints = relationship("Checkpoint", back_populates="training_job")

class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    step = Column(Integer)
    path = Column(String)
    metrics = Column(JSON, nullable=True)
    is_best = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="checkpoints")
EOF
```

## 7. Create the Core Job Queue System

```bash
# Create core/queue.py
cat > core/queue.py << 'EOF'
import redis
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

class JobQueue:
    """
    Redis-based job queue for training jobs
    """
    
    def __init__(self):
        """Initialize Redis connection"""
        self.redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.queue_key = "unslothui:job_queue"
        self.running_key = "unslothui:running_jobs"
        self.job_data_prefix = "unslothui:job:"
    
    async def enqueue_job(self, job_id: int, job_data: Dict[str, Any]) -> str:
        """
        Add a job to the queue
        
        Args:
            job_id: Job ID
            job_data: Additional job data
        
        Returns:
            Queue ID
        """
        queue_id = f"job:{job_id}:{datetime.now().timestamp()}"
        
        # Store job data
        self.redis.set(
            f"{self.job_data_prefix}{queue_id}",
            json.dumps(job_data)
        )
        
        # Add to queue
        self.redis.rpush(self.queue_key, queue_id)
        
        logger.info(f"Job {job_id} added to queue with ID {queue_id}")
        return queue_id
    
    async def dequeue_job(self) -> Optional[Dict[str, Any]]:
        """
        Get the next job from the queue
        
        Returns:
            Job data or None if queue is empty
        """
        # Check if we're below max concurrent jobs
        running_count = self.redis.scard(self.running_key)
        if running_count >= settings.MAX_CONCURRENT_JOBS:
            return None
        
        # Get next job
        queue_id = self.redis.lpop(self.queue_key)
        if not queue_id:
            return None
        
        # Get job data
        job_data_key = f"{self.job_data_prefix}{queue_id}"
        job_data_str = self.redis.get(job_data_key)
        
        if not job_data_str:
            logger.error(f"Job data not found for {queue_id}")
            return None
        
        try:
            job_data = json.loads(job_data_str)
            job_data["queue_id"] = queue_id
            
            # Mark as running
            self.redis.sadd(self.running_key, queue_id)
            
            return job_data
        
        except Exception as e:
            logger.exception(f"Error parsing job data for {queue_id}")
            return None
    
    async def complete_job(self, queue_id: str) -> bool:
        """
        Mark a job as completed
        
        Args:
            queue_id: Queue ID
        
        Returns:
            Success status
        """
        # Remove from running set
        removed = self.redis.srem(self.running_key, queue_id)
        
        # Clean up job data
        self.redis.delete(f"{self.job_data_prefix}{queue_id}")
        
        return removed > 0
    
    async def fail_job(self, queue_id: str, error: str) -> bool:
        """
        Mark a job as failed
        
        Args:
            queue_id: Queue ID
            error: Error message
        
        Returns:
            Success status
        """
        # Remove from running set
        removed = self.redis.srem(self.running_key, queue_id)
        
        # Update job data with error
        job_data_key = f"{self.job_data_prefix}{queue_id}"
        job_data_str = self.redis.get(job_data_key)
        
        if job_data_str:
            try:
                job_data = json.loads(job_data_str)
                job_data["error"] = error
                self.redis.set(job_data_key, json.dumps(job_data))
            except:
                pass
        
        return removed > 0
    
    async def get_queue_length(self) -> int:
        """
        Get the number of jobs in the queue
        
        Returns:
            Queue length
        """
        return self.redis.llen(self.queue_key)
    
    async def get_running_jobs(self) -> List[str]:
        """
        Get a list of running job IDs
        
        Returns:
            List of queue IDs
        """
        return self.redis.smembers(self.running_key)
    
    async def clear_queue(self) -> int:
        """
        Clear the job queue
        
        Returns:
            Number of jobs removed
        """
        length = await self.get_queue_length()
        self.redis.delete(self.queue_key)
        return length

class MetricsManager:
    """
    Manager for real-time training metrics
    """
    
    def __init__(self, job_id: int):
        """
        Initialize metrics manager for a specific job
        
        Args:
            job_id: Training job ID
        """
        self.job_id = job_id
        self.redis = redis.Redis.from_url(settings.REDIS_URL)
        self.metrics_key = f"unslothui:metrics:{job_id}"
        self.clients = set()
    
    async def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update metrics and broadcast to clients
        
        Args:
            metrics: New metrics values
        """
        # Store metrics in Redis
        self.redis.set(self.metrics_key, json.dumps(metrics))
        
        # Broadcast to connected clients
        if self.clients:
            message = json.dumps({
                "type": "metrics",
                "job_id": self.job_id,
                "data": metrics,
                "timestamp": datetime.now().isoformat()
            })
            
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send_text(message)
                except Exception:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected
    
    async def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest metrics values
        
        Returns:
            Latest metrics
        """
        metrics_str = self.redis.get(self.metrics_key)
        if metrics_str:
            try:
                return json.loads(metrics_str)
            except:
                return {}
        return {}
    
    async def register_client(self, websocket):
        """
        Register a new client for metrics updates
        
        Args:
            websocket: WebSocket connection to client
        """
        self.clients.add(websocket)
        
        # Send current metrics
        metrics = await self.get_latest_metrics()
        if metrics:
            await websocket.send_text(json.dumps({
                "type": "metrics",
                "job_id": self.job_id,
                "data": metrics,
                "timestamp": datetime.now().isoformat()
            }))
    
    async def unregister_client(self, websocket):
        """
        Unregister a client
        
        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.clients:
            self.clients.remove(websocket)

class LogManager:
    """
    Manager for streaming training logs
    """
    
    def __init__(self, job_id: int):
        """
        Initialize log manager for a specific job
        
        Args:
            job_id: Training job ID
        """
        self.job_id = job_id
        self.redis = redis.Redis.from_url(settings.REDIS_URL)
        self.log_key = f"unslothui:logs:{job_id}"
        self.clients = set()
    
    async def add_log(self, message: str):
        """
        Add a log message and broadcast to clients
        
        Args:
            message: Log message
        """
        timestamp = datetime.now().isoformat()
        
        # Store in Redis (limited list)
        log_entry = json.dumps({
            "message": message,
            "timestamp": timestamp
        })
        
        # Add to limited list (last 1000 messages)
        self.redis.lpush(self.log_key, log_entry)
        self.redis.ltrim(self.log_key, 0, 999)
        
        # Broadcast to connected clients
        if self.clients:
            ws_message = json.dumps({
                "type": "log",
                "job_id": self.job_id,
                "message": message,
                "timestamp": timestamp
            })
            
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send_text(ws_message)
                except Exception:
                    disconnected.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected
    
    async def get_logs(self, limit: int = 100) -> List[Dict[str, str]]:
        """
        Get recent log messages
        
        Args:
            limit: Maximum number of messages to return
        
        Returns:
            List of log messages with timestamps
        """
        log_entries = self.redis.lrange(self.log_key, 0, limit - 1)
        logs = []
        
        for entry in log_entries:
            try:
                logs.append(json.loads(entry))
            except:
                pass
        
        return logs
    
    async def register_client(self, websocket):
        """
        Register a new client for log updates
        
        Args:
            websocket: WebSocket connection to client
        """
        self.clients.add(websocket)
        
        # Send recent logs
        logs = await self.get_logs(100)
        if logs:
            await websocket.send_text(json.dumps({
                "type": "log_history",
                "job_id": self.job_id,
                "logs": logs
            }))
    
    async def unregister_client(self, websocket):
        """
        Unregister a client
        
        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.clients:
            self.clients.remove(websocket)
EOF
```

## 8. Package as GitHub Repository

Once you've created all these files, you can finalize the repository:

```bash
# Initialize the database
python -c "from core.database import Base, engine; Base.metadata.create_all(bind=engine)"

# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Make initial commit
git add .
git commit -m "Initial commit of UnslothUI backend"

# Optional: Create a ZIP of the repository
cd ..
zip -r unslothui.zip unslothui
```

## Additional Files

To complete the repository, you still need to create these important files:

1. Complete API endpoints (`auth.py`, `models.py`, `datasets.py`, `training.py`, `evaluation.py`)
2. Service implementations (`model_service.py`, `dataset_service.py`, `training_service.py`, `evaluation_service.py`)
3. Integration implementations (`unsloth.py`, `huggingface.py`)
4. Worker implementation (`training_worker.py`)
5. Test files (`conftest.py`, `test_auth.py`, etc.)

These were provided in my earlier response. I recommend copying those files to their appropriate locations within the repository structure.

## Deployment Instructions

For deploying this application:

1. **Local Development**:
   ```bash
   # Set up a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Create .env file
   cp .env.example .env
   # Edit .env with your settings
   
   # Initialize database
   alembic upgrade head
   
   # Run the API server
   python main.py
   
   # Run the worker (in a separate terminal)
   python -m workers.training_worker
   ```

2. **Docker Deployment**:
   ```bash
   # Start all services
   docker-compose up -d
   
   # Initialize database (first time only)
   docker-compose exec api alembic upgrade head
   
   # View logs
   docker-compose logs -f
   ```

This completes the setup and packaging of the UnslothUI GitHub repository. The structure follows best practices for a production-ready Python application with a clean architecture, proper separation of concerns, and comprehensive configuration management.
