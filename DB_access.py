from app.db.session import get_db
from app.db.models import ComplaintEmbedding

with get_db() as db:
    print(db.query(ComplaintEmbedding).count())
    for row in db.query(ComplaintEmbedding).limit(10):
        print(row.complaint_id, row.product, row.issue, row.company)