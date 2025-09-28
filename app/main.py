import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .core.config import get_settings
from .db.session import Base, get_engine, get_sessionmaker
from .models.user import User
from .routers import auth as auth_router
from .routers import predictions as predictions_router
from .routers import stats as stats_router
from .routers import model_info as model_info_router
from .services.auth import get_password_hash
from .services import inference


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    os.makedirs(settings.media_root, exist_ok=True)
    app.mount("/media", StaticFiles(directory=settings.media_root), name="media")

    # DB init
    engine = get_engine(settings.database_url)
    Base.metadata.create_all(bind=engine)

    # Seed admin if not exists
    SessionLocal = get_sessionmaker(settings.database_url)
    with SessionLocal() as db:  # type: Session
        if not db.query(User).filter(User.username == settings.admin_username).first():
            db.add(
                User(
                    username=settings.admin_username,
                    password_hash=get_password_hash(settings.admin_password),
                )
            )
            db.commit()

    # Load model on startup (fast subsequent predictions)
    @app.on_event("startup")
    def _load_model():
        model_path = os.path.join(settings.models_path, settings.model_file)
        inference.ensure_loaded(model_path)

    app.include_router(auth_router.router)
    app.include_router(predictions_router.router)
    app.include_router(stats_router.router)
    app.include_router(model_info_router.router)
    return app


app = create_app()
