from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.config import settings

app = FastAPI(
    title="Conversational AI Agent API",
    description=(
        "FastAPI Backend for Tool-Calling Conversational AI Agent with multi-turn conversation memory, "
        "Web Search, Calculator, and Datetime tools returning structured JSON outputs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router)


@app.get("/", summary="Root Endpoint")
async def root():
    """Root status response."""
    return {
        "message": "Conversational AI Agent API is running.",
        "docs": "/docs",
        "health": "/health",
        "chat_endpoint": "/api/v1/chat"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True
    )
