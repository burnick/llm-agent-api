"""
Main FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_current_config, validate_startup_config
from app.logging import setup_logging, get_logger, LoggingMiddleware, ErrorLoggingMiddleware


# Global configuration instance
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global config
    
    # Startup
    try:
        # Validate configuration at startup
        validate_startup_config()
        config = get_current_config()
        
        # Setup structured logging
        setup_logging(
            environment=config.environment,
            log_level=config.api.log_level,
            enable_json=(config.environment.value == "production")
        )
        
        logger = get_logger("main")
        logger.info(
            "Starting LLM Agent API",
            extra={
                "environment": config.environment.value,
                "debug_mode": config.debug,
                "llm_provider": config.llm.default_provider.value,
                "api_host": config.api.host,
                "api_port": config.api.port,
                "log_level": config.api.log_level.value,
                "startup": True,
            }
        )
        
    except Exception as e:
        # Use basic logging if structured logging setup fails
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Failed to initialize application: {e}", exc_info=e)
        raise
    
    yield
    
    # Shutdown
    logger = get_logger("main")
    logger.info("Shutting down LLM Agent API", extra={"shutdown": True})


# Create FastAPI application instance
app = FastAPI(
    title="LLM Agent API",
    description="FastAPI backend with LLM and agent capabilities",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.on_event("startup")
async def setup_middleware():
    """Setup middleware after configuration is loaded."""
    global config
    if config:
        # Add logging middleware first (to capture all requests)
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(ErrorLoggingMiddleware)
        
        # Add CORS middleware with configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    global config
    logger = get_logger("endpoints")
    
    response_data = {
        "message": "LLM Agent API",
        "version": "0.1.0",
        "status": "running",
        "environment": config.environment if config else "unknown",
        "debug": config.debug if config else False,
    }
    
    logger.info("Root endpoint accessed", extra={"endpoint": "root"})
    return response_data


@app.get("/health")
async def health_check():
    """Health check endpoint with configuration status."""
    global config
    logger = get_logger("endpoints")
    
    health_status = {
        "status": "healthy",
        "service": "llm-agent-api",
        "version": "0.1.0",
    }
    
    if config:
        health_status.update({
            "environment": config.environment,
            "llm_provider": config.llm.default_provider,
            "debug_mode": config.debug,
        })
    
    logger.info(
        "Health check performed",
        extra={
            "endpoint": "health",
            "status": health_status["status"],
            "environment": health_status.get("environment"),
        }
    )
    
    return health_status


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    import uvicorn
    
    # Load configuration for development server
    try:
        validate_startup_config()
        dev_config = get_current_config()
        
        uvicorn.run(
            "app.main:app",
            host=dev_config.api.host,
            port=dev_config.api.port,
            reload=dev_config.debug,
            log_level=dev_config.api.log_level.value.lower(),
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        exit(1)