"""
db/database.py
SQLAlchemy engine, session factory, and DB initialisation.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from db.models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
DB_PATH = _ROOT / "hr_valais_prototype.db"
DATA_DIR = _ROOT / "data"

# ---------------------------------------------------------------------------
# Engine & session
# ---------------------------------------------------------------------------
_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            echo=False,
        )

        @event.listens_for(_engine, "connect")
        def _set_wal(dbapi_con, _):
            dbapi_con.execute("PRAGMA journal_mode=WAL")
            dbapi_con.execute("PRAGMA foreign_keys=ON")
    return _engine


def _get_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionFactory


@contextmanager
def get_session():
    """
    Context-manager session. Use as:
        with get_session() as s:
            s.query(...)
    Commits on success, rolls back on exception, always closes.
    """
    factory = _get_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Initialise DB (idempotent)
# ---------------------------------------------------------------------------
def init_db(force_reseed: bool = False) -> None:
    """
    Create all tables (if not present) then seed if DB is empty.
    Designed to be called once at app startup; safe to call repeatedly.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("Tables ensured.")

    with get_session() as session:
        from db.models import Firm
        count = session.query(Firm).count()

    if count == 0 or force_reseed:
        logger.info("DB empty — running cold-start seed.")
        from db.seed import run_seed
        run_seed()
    else:
        logger.info(f"DB already seeded ({count} firms found). Skipping.")
