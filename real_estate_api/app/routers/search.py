'''
    The router for search-related endpoints
'''

# real_estate_project/real_estate_api/app/routers/search.py

from fastapi import APIRouter

router = APIRouter(
    prefix="/search",
    tags=["search"],
)

@router.get("/")
def search_properties():
    return {"message": "Search functionality to be implemented"}


