from .routes import router
from .company import router as company_router

router.include_router(company_router)

__all__ = ["router"]
