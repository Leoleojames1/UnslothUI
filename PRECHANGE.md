# UnslothUI Backend Implementation for Hugging Face Integration

I'll create a comprehensive Python backend for UnslothUI that integrates with Hugging Face's ecosystem. This implementation focuses on providing a robust API server, training service, storage layer, and authentication service as outlined in your architecture specification.

## Main Application Structure

First, let's establish the main application entry point:

```python
# main.py
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

## Core Configuration

Let's set up the core configuration module:

```python
# core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "UnslothUI"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # API settings
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]  # Frontend URL
    
    # Authentication settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
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
```

## Database Setup

Next, let's set up the database connection and models:

```python
# core/database.py
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
```

## Security Implementation

Let's implement JWT-based authentication:

```python
# core/security.py
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
```

## Model Definitions

Let's define the database models:

```python
# models/user.py
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.sql import func
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

# models/dataset.py
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

# models/model.py
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

# models/job.py
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
```

## API Endpoints

Now, let's implement the API endpoints:

```python
# api/auth.py
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Any
from pydantic import BaseModel

from core.database import get_db
from core.security import verify_password, create_access_token, create_refresh_token
from models.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
) -> Any:
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": str(user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str, db: Session = Depends(get_db)
) -> Any:
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        is_refresh = payload.get("refresh", False)
        
        if not user_id or not is_refresh:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        new_refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=Token)
async def register_user(
    user_data: UserCreate, db: Session = Depends(get_db)
) -> Any:
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.username == user_data.username) | (User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered",
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create tokens
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(new_user.id)}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": str(new_user.id)})
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
```

Now let's implement the models API:

```python
# api/models.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.model import ModelConfig
from services.model_service import ModelService
from integrations.huggingface import HuggingFaceClient

router = APIRouter(prefix="/models", tags=["Models"])

class ModelConfigCreate(BaseModel):
    name: str
    base_model_id: str
    description: Optional[str] = None
    lora_config: dict
    training_config: dict
    quantization_config: dict

class ModelConfigResponse(BaseModel):
    id: int
    name: str
    base_model_id: str
    description: Optional[str] = None
    lora_config: dict
    training_config: dict
    quantization_config: dict
    created_at: str
    updated_at: Optional[str] = None
    
    class Config:
        from_attributes = True

class HuggingFaceModelInfo(BaseModel):
    model_id: str
    name: str
    description: Optional[str] = None
    downloads: Optional[int] = None
    likes: Optional[int] = None
    tags: List[str] = []

@router.get("/available", response_model=List[HuggingFaceModelInfo])
async def get_available_models(
    query: Optional[str] = None,
    filter_tags: Optional[str] = None,
    sort_by: Optional[str] = "downloads",
    limit: int = 20,
    user: User = Depends(get_current_user)
):
    """
    Get list of available base models from HuggingFace Hub
    """
    hf_client = HuggingFaceClient()
    tags = filter_tags.split(",") if filter_tags else None
    models = await hf_client.search_models(query, tags, sort_by, limit)
    return models

@router.get("/{model_id}", response_model=HuggingFaceModelInfo)
async def get_model_details(
    model_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get details for a specific base model from HuggingFace Hub
    """
    hf_client = HuggingFaceClient()
    model = await hf_client.get_model_info(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return model

@router.get("/configurations", response_model=List[ModelConfigResponse])
async def get_user_model_configurations(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get all model configurations created by the user
    """
    configurations = db.query(ModelConfig).filter(ModelConfig.user_id == user.id).all()
    return configurations

@router.post("/configurations", response_model=ModelConfigResponse)
async def create_model_configuration(
    config: ModelConfigCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Create a new model configuration
    """
    model_service = ModelService(db)
    new_config = await model_service.create_configuration(config, user.id)
    return new_config

@router.get("/configurations/{config_id}", response_model=ModelConfigResponse)
async def get_model_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get a specific model configuration
    """
    config = db.query(ModelConfig).filter(
        ModelConfig.id == config_id,
        ModelConfig.user_id == user.id
    ).first()
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    
    return config

@router.put("/configurations/{config_id}", response_model=ModelConfigResponse)
async def update_model_configuration(
    config_id: int,
    config_update: ModelConfigCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Update an existing model configuration
    """
    model_service = ModelService(db)
    updated_config = await model_service.update_configuration(config_id, config_update, user.id)
    if not updated_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    
    return updated_config

@router.delete("/configurations/{config_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Delete a model configuration
    """
    model_service = ModelService(db)
    success = await model_service.delete_configuration(config_id, user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with ID {config_id} not found"
        )
    
    return {"status": "success"}
```

Let's implement the training API with WebSocket support:

```python
# api/training.py
from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from core.database import get_db
from core.security import get_current_user
from models.user import User
from models.job import TrainingJob, JobStatus, Checkpoint
from services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["Training"])

class TrainingJobCreate(BaseModel):
    name: str
    model_config_id: int
    dataset_id: int
    device: Optional[str] = None  # If None, use default device

class TrainingJobResponse(BaseModel):
    id: int
    name: str
    user_id: int
    model_config_id: int
    dataset_id: int
    status: str
    device: str
    current_step: int
    total_steps: int
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    
    class Config:
        from_attributes = True

class CheckpointResponse(BaseModel):
    id: int
    training_job_id: int
    step: int
    path: str
    metrics: Optional[Dict[str, Any]] = None
    is_best: bool
    created_at: str
    
    class Config:
        from_attributes = True

@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    job: TrainingJobCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Create and start a new training job
    """
    training_service = TrainingService(db)
    new_job = await training_service.create_job(job, user.id)
    
    # Start the job asynchronously
    await training_service.queue_job(new_job.id)
    
    return new_job

@router.get("/jobs", response_model=List[TrainingJobResponse])
async def get_training_jobs(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get all training jobs for the user
    """
    query = db.query(TrainingJob).filter(TrainingJob.user_id == user.id)
    
    if status:
        try:
            job_status = JobStatus[status.upper()]
            query = query.filter(TrainingJob.status == job_status)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}"
            )
    
    # Calculate pagination
    total = query.count()
    query = query.order_by(TrainingJob.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    jobs = query.all()
    
    # Add pagination metadata
    return {
        "items": jobs,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size
    }

@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get details for a specific training job
    """
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job with ID {job_id} not found"
        )
    
    return job

@router.post("/jobs/{job_id}/stop")
async def stop_training_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Stop a running training job
    """
    training_service = TrainingService(db)
    success = await training_service.stop_job(job_id, user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job with ID {job_id} not found or not running"
        )
    
    return {"status": "success", "message": "Job stopping initiated"}

@router.get("/jobs/{job_id}/checkpoints", response_model=List[CheckpointResponse])
async def get_job_checkpoints(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get all checkpoints for a training job
    """
    # Verify job exists and belongs to user
    job = db.query(TrainingJob).filter(
        TrainingJob.id == job_id,
        TrainingJob.user_id == user.id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job with ID {job_id} not found"
        )
    
    checkpoints = db.query(Checkpoint).filter(
        Checkpoint.training_job_id == job_id
    ).order_by(Checkpoint.step).all()
    
    return checkpoints

@router.post("/jobs/{job_id}/checkpoint")
async def create_manual_checkpoint(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Create a manual checkpoint for a running job
    """
    training_service = TrainingService(db)
    checkpoint = await training_service.create_manual_checkpoint(job_id, user.id)
    
    if not checkpoint:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not create checkpoint for job ID {job_id}"
        )
    
    return checkpoint

@router.websocket("/ws/jobs/{job_id}/metrics")
async def websocket_metrics(
    websocket: WebSocket,
    job_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time training metrics
    """
    await websocket.accept()
    
    try:
        # Verify token and check job access
        user = await get_current_user(token, db)
        job = db.query(TrainingJob).filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == user.id
        ).first()
        
        if not job:
            await websocket.close(code=4004, reason="Job not found or access denied")
            return
        
        training_service = TrainingService(db)
        metrics_manager = await training_service.get_metrics_manager(job_id)
        
        # Register this client
        await metrics_manager.register_client(websocket)
        
        try:
            while True:
                # Keep connection alive, actual sending is done by metrics manager
                await websocket.receive_text()
        except WebSocketDisconnect:
            # Unregister when client disconnects
            await metrics_manager.unregister_client(websocket)
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))

@router.websocket("/ws/jobs/{job_id}/logs")
async def websocket_logs(
    websocket: WebSocket,
    job_id: int,
    token: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for streaming training logs
    """
    await websocket.accept()
    
    try:
        # Verify token and check job access
        user = await get_current_user(token, db)
        job = db.query(TrainingJob).filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == user.id
        ).first()
        
        if not job:
            await websocket.close(code=4004, reason="Job not found or access denied")
            return
        
        training_service = TrainingService(db)
        log_manager = await training_service.get_log_manager(job_id)
        
        # Register this client
        await log_manager.register_client(websocket)
        
        try:
            while True:
                # Keep connection alive, actual sending is done by log manager
                await websocket.receive_text()
        except WebSocketDisconnect:
            # Unregister when client disconnects
            await log_manager.unregister_client(websocket)
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))
```

## Unsloth Integration

Let's implement the Unsloth integration for LLM fine-tuning:

```python
# integrations/unsloth.py
from typing import Dict, Any, Optional, List, Tuple
import os
import json
import torch
from pathlib import Path
import logging
from core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

class UnslothTrainer:
    """
    Integration with Unsloth for efficient LLM fine-tuning
    """
    
    def __init__(
        self,
        base_model_id: str,
        output_dir: str,
        lora_config: Dict[str, Any],
        training_config: Dict[str, Any],
        quantization_config: Dict[str, Any],
        device: str = "cuda:0",
        metrics_callback = None,
        log_callback = None
    ):
        """
        Initialize Unsloth trainer with configurations
        
        Args:
            base_model_id: HuggingFace model ID for the base model
            output_dir: Directory to save outputs
            lora_config: LoRA parameters configuration
            training_config: Training hyperparameters
            quantization_config: Quantization settings
            device: Training device (cuda:0, cuda:1, cpu, etc.)
            metrics_callback: Callback function for metrics updates
            log_callback: Callback function for log updates
        """
        self.base_model_id = base_model_id
        self.output_dir = output_dir
        self.device = device
        self.lora_config = lora_config
        self.training_config = training_config
        self.quantization_config = quantization_config
        self.metrics_callback = metrics_callback
        self.log_callback = log_callback
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Save configurations
        self._save_config()
    
    def _save_config(self):
        """Save configuration to output directory"""
        config = {
            "base_model_id": self.base_model_id,
            "lora_config": self.lora_config,
            "training_config": self.training_config,
            "quantization_config": self.quantization_config,
            "device": self.device
        }
        
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    async def load_model(self):
        """
        Load the base model with Unsloth optimizations
        """
        try:
            # Import here to avoid dependency issues when not using this module
            from unsloth import FastLanguageModel
            
            # Log message
            if self.log_callback:
                await self.log_callback(f"Loading model {self.base_model_id}...")
            
            # Extract configuration parameters
            max_seq_length = self.training_config.get("max_seq_length", 2048)
            load_in_4bit = self.quantization_config.get("load_in_4bit", True)
            load_in_8bit = self.quantization_config.get("load_in_8bit", False)
            
            # Determine dtype
            dtype_str = self.training_config.get("dtype", "auto")
            if dtype_str == "float16":
                dtype = torch.float16
            elif dtype_str == "bfloat16":
                dtype = torch.bfloat16
            elif dtype_str == "float32":
                dtype = torch.float32
            else:  # auto
                dtype = None
            
            # Load model and tokenizer with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_id,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            
            # Configure LoRA
            lora_rank = self.lora_config.get("rank", 16)
            lora_alpha = self.lora_config.get("alpha", 16)
            lora_dropout = self.lora_config.get("dropout", 0.05)
            target_modules = self.lora_config.get("target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            
            # Apply LoRA
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_rank,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            # Log success
            if self.log_callback:
                await self.log_callback(f"Successfully loaded model and applied LoRA configuration")
            
            return True
        
        except Exception as e:
            logger.exception("Error loading model")
            if self.log_callback:
                await self.log_callback(f"Error loading model: {str(e)}")
            return False
    
    async def prepare_dataset(self, dataset_path, format_type="json"):
        """
        Prepare dataset for training
        
        Args:
            dataset_path: Path to the dataset file
            format_type: Format of the dataset (json, csv, txt)
        
        Returns:
            Prepared dataset
        """
        try:
            from datasets import load_dataset
            
            if self.log_callback:
                await self.log_callback(f"Loading dataset from {dataset_path}...")
            
            # Load dataset based on format
            if format_type.lower() == "json" or format_type.lower() == "jsonl":
                dataset = load_dataset("json", data_files={"train": dataset_path})["train"]
            elif format_type.lower() == "csv":
                dataset = load_dataset("csv", data_files={"train": dataset_path})["train"]
            elif format_type.lower() == "text" or format_type.lower() == "txt":
                dataset = load_dataset("text", data_files={"train": dataset_path})["train"]
            else:
                raise ValueError(f"Unsupported dataset format: {format_type}")
            
            # Apply formatting if needed
            text_column = self.training_config.get("text_column", "text")
            
            # Apply dataset preprocessing if needed
            # ... additional preprocessing logic based on configuration ...
            
            if self.log_callback:
                await self.log_callback(f"Dataset loaded with {len(dataset)} examples")
            
            return dataset
        
        except Exception as e:
            logger.exception("Error preparing dataset")
            if self.log_callback:
                await self.log_callback(f"Error preparing dataset: {str(e)}")
            return None
    
    async def train(self, dataset):
        """
        Run training process with configured parameters
        
        Args:
            dataset: Prepared dataset for training
        
        Returns:
            Training results
        """
        try:
            from transformers import TrainingArguments
            from trl import SFTTrainer, SFTConfig
            
            if self.log_callback:
                await self.log_callback("Starting training process...")
            
            # Extract training parameters
            learning_rate = self.training_config.get("learning_rate", 2e-4)
            num_train_epochs = self.training_config.get("num_train_epochs", 3)
            per_device_train_batch_size = self.training_config.get("per_device_train_batch_size", 4)
            gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 1)
            optim = self.training_config.get("optimizer", "adamw_torch")
            weight_decay = self.training_config.get("weight_decay", 0.01)
            max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
            warmup_ratio = self.training_config.get("warmup_ratio", 0.03)
            lr_scheduler_type = self.training_config.get("lr_scheduler_type", "cosine")
            logging_steps = self.training_config.get("logging_steps", 10)
            save_steps = self.training_config.get("save_steps", 100)
            eval_steps = self.training_config.get("eval_steps", 100)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                save_steps=save_steps,
                logging_steps=logging_steps,
                save_total_limit=3,
                optim=optim,
                fp16=(self.training_config.get("dtype", "auto") == "float16"),
                bf16=(self.training_config.get("dtype", "auto") == "bfloat16"),
                report_to="none",  # We'll handle metrics ourselves
            )
            
            # Custom callback to capture metrics
            class MetricsCallback:
                def __init__(self, metrics_callback):
                    self.metrics_callback = metrics_callback
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and self.metrics_callback:
                        asyncio.create_task(self.metrics_callback(logs))
            
            # Setup trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=training_args,
                peft_config=None,  # We already applied PEFT config
                callbacks=[MetricsCallback(self.metrics_callback)] if self.metrics_callback else None,
            )
            
            # Run training
            if self.log_callback:
                await self.log_callback("Training started...")
            
            train_result = trainer.train()
            
            # Save final model
            trainer.save_model(os.path.join(self.output_dir, "final"))
            
            # Save metrics
            metrics = train_result.metrics
            with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            
            if self.log_callback:
                await self.log_callback(f"Training completed. Final loss: {metrics.get('train_loss', 'N/A')}")
            
            return {
                "success": True,
                "metrics": metrics,
                "output_dir": self.output_dir
            }
        
        except Exception as e:
            logger.exception("Error during training")
            if self.log_callback:
                await self.log_callback(f"Error during training: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def save_checkpoint(self, step=None):
        """
        Save a checkpoint of the current model
        
        Args:
            step: Current training step (if None, use timestamp)
        
        Returns:
            Path to the saved checkpoint
        """
        if not self.model:
            return None
        
        try:
            from datetime import datetime
            
            # Create checkpoint directory
            if step is not None:
                checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"step_{step}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"manual_{timestamp}")
            
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            if self.log_callback:
                await self.log_callback(f"Checkpoint saved to {checkpoint_dir}")
            
            return checkpoint_dir
        
        except Exception as e:
            logger.exception("Error saving checkpoint")
            if self.log_callback:
                await self.log_callback(f"Error saving checkpoint: {str(e)}")
            return None
    
    async def export_model(self, export_format="huggingface", export_dir=None):
        """
        Export the fine-tuned model
        
        Args:
            export_format: Format to export (huggingface, gguf)
            export_dir: Directory to save the exported model
        
        Returns:
            Path to the exported model
        """
        if not self.model:
            return None
        
        try:
            if export_dir is None:
                export_dir = os.path.join(self.output_dir, "exported")
            
            os.makedirs(export_dir, exist_ok=True)
            
            if self.log_callback:
                await self.log_callback(f"Exporting model in {export_format} format...")
            
            if export_format == "huggingface":
                # Export to HuggingFace format
                from peft import PeftModel
                
                # Save adapter
                self.model.save_pretrained(export_dir)
                self.tokenizer.save_pretrained(export_dir)
                
                # Save configuration info
                with open(os.path.join(export_dir, "unsloth_info.json"), "w") as f:
                    json.dump({
                        "base_model": self.base_model_id,
                        "lora_config": self.lora_config,
                        "exported_at": datetime.now().isoformat(),
                    }, f, indent=2)
                
                if self.log_callback:
                    await self.log_callback(f"Model exported to {export_dir}")
                
                return export_dir
            
            elif export_format == "gguf":
                # Export to GGUF format for llama.cpp
                if self.log_callback:
                    await self.log_callback("GGUF export requires additional steps...")
                
                # First, merge weights to get full model
                merged_dir = os.path.join(export_dir, "merged")
                os.makedirs(merged_dir, exist_ok=True)
                
                # Get merged model
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(merged_dir)
                self.tokenizer.save_pretrained(merged_dir)
                
                # Indicate that additional step is needed
                with open(os.path.join(export_dir, "gguf_export_info.txt"), "w") as f:
                    f.write(
                        "To convert this model to GGUF format, use the llama.cpp conversion script:\n"
                        "python3 -m llama_cpp.convert_hf_to_gguf path/to/merged/model --outfile model.gguf\n\n"
                        "This requires the llama-cpp-python package to be installed."
                    )
                
                if self.log_callback:
                    await self.log_callback(
                        f"Model prepared for GGUF export at {merged_dir}. "
                        "Additional conversion step required."
                    )
                
                return merged_dir
            
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
        
        except Exception as e:
            logger.exception("Error exporting model")
            if self.log_callback:
                await self.log_callback(f"Error exporting model: {str(e)}")
            return None
```

## Hugging Face Integration

Let's implement the Hugging Face integration:

```python
# integrations/huggingface.py
from typing import List, Dict, Optional, Any
import aiohttp
import os
from core.config import settings
import logging

# Setup logging
logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """
    Client for interacting with the Hugging Face Hub API
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace client with API token
        
        Args:
            token: HuggingFace API token (if None, use from settings)
        """
        self.token = token or settings.HF_TOKEN
        self.base_url = "https://huggingface.co/api"
        self.headers = {
            "Authorization": f"Bearer {self.token}" if self.token else ""
        }
    
    async def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort: str = "downloads",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for models on HuggingFace Hub
        
        Args:
            query: Search query string
            tags: List of tags to filter by
            sort: Sort field (downloads, likes, modified)
            limit: Maximum number of results
        
        Returns:
            List of model information
        """
        try:
            params = {
                "limit": limit,
                "sort": sort,
                "full": "true"
            }
            
            if query:
                params["search"] = query
            
            if tags:
                params["filter"] = ",".join(tags)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    params=params,
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format response
                        models = []
                        for model in data:
                            models.append({
                                "model_id": model.get("id", ""),
                                "name": model.get("modelId", model.get("id", "")),
                                "description": model.get("description", ""),
                                "downloads": model.get("downloads", 0),
                                "likes": model.get("likes", 0),
                                "tags": model.get("tags", [])
                            })
                        
                        return models
                    else:
                        logger.error(f"Error searching models: {response.status}")
                        return []
        
        except Exception as e:
            logger.exception("Error searching HuggingFace models")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model
        
        Args:
            model_id: HuggingFace model ID
        
        Returns:
            Model information or None if not found
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models/{model_id}",
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        model = await response.json()
                        
                        return {
                            "model_id": model.get("id", ""),
                            "name": model.get("modelId", model.get("id", "")),
                            "description": model.get("description", ""),
                            "downloads": model.get("downloads", 0),
                            "likes": model.get("likes", 0),
                            "tags": model.get("tags", [])
                        }
                    else:
                        logger.error(f"Error getting model info: {response.status}")
                        return None
        
        except Exception as e:
            logger.exception(f"Error getting model info for {model_id}")
            return None
    
    async def download_model(
        self,
        model_id: str,
        target_dir: str,
        revision: Optional[str] = None
    ) -> bool:
        """
        Download a model from HuggingFace Hub
        
        Args:
            model_id: HuggingFace model ID
            target_dir: Directory to save the model
            revision: Specific revision to download
        
        Returns:
            Success status
        """
        try:
            # Use Hugging Face Hub API to download
            from huggingface_hub import snapshot_download
            
            # Create target directory
            os.makedirs(target_dir, exist_ok=True)
            
            # Download model
            snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                revision=revision,
                token=self.token,
                local_dir_use_symlinks=False
            )
            
            return True
        
        except Exception as e:
            logger.exception(f"Error downloading model {model_id}")
            return False
    
    async def upload_model(
        self,
        model_path: str,
        repo_id: str,
        commit_message: str = "Upload from UnslothUI",
        private: bool = False
    ) -> bool:
        """
        Upload a model to HuggingFace Hub
        
        Args:
            model_path: Path to the model files
            repo_id: Target repository ID (username/model-name)
            commit_message: Commit message
            private: Whether to create a private repository
        
        Returns:
            Success status
        """
        try:
            if not self.token:
                logger.error("HuggingFace token is required for model upload")
                return False
            
            # Use Hugging Face Hub API to upload
            from huggingface_hub import HfApi
            
            api = HfApi(token=self.token)
            
            # Create repository if it doesn't exist
            try:
                api.create_repo(
                    repo_id=repo_id,
                    private=private,
                    repo_type="model",
                    exist_ok=True
                )
            except Exception as e:
                logger.error(f"Error creating repository: {str(e)}")
                return False
            
            # Upload model files
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                commit_message=commit_message
            )
            
            return True
        
        except Exception as e:
            logger.exception(f"Error uploading model to {repo_id}")
            return False
```

## Job Queue Implementation

Let's implement the job queue system:

```python
# core/queue.py
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
```

## Requirements File

Let's create a requirements.txt file for the project:

```
# requirements.txt
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
```

## Implementation Summary

This backend implementation provides a robust foundation for the UnslothUI application, with:

1. **FastAPI Server**: High-performance API with WebSocket support for real-time metrics
2. **Unsloth Integration**: Efficient fine-tuning of LLMs with lower memory requirements
3. **Hugging Face Integration**: Seamless model discovery, download, and upload
4. **Job Queue**: Redis-based queue for managing long-running training jobs
5. **Database Models**: SQLAlchemy models for users, datasets, models, and training jobs
6. **Authentication**: JWT-based authentication with token refresh

To use this backend:
1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables or create a .env file
3. Initialize the database: `alembic upgrade head`
4. Start the server: `python main.py`

This implementation follows the architecture specifications from your document and provides a solid foundation that can be extended with additional features as needed.
