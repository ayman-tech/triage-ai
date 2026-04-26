from .routes import router
from .company import router as company_router
from .policies import router as policies_router

router.include_router(company_router)
router.include_router(policies_router)

__all__ = ["router"]
