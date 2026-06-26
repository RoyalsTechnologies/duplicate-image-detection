"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-23
"""

from alembic import op
import sqlalchemy as sa
from geoalchemy2 import Geometry
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ENUM

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

report_category = ENUM(
    "refuse_dump",
    "blocked_drain",
    "flooding",
    "pothole",
    "pollution",
    "broken_public_facility",
    "sanitation",
    "other",
    name="report_category",
    create_type=False,
)
duplicate_status = ENUM(
    "new",
    "duplicate",
    "possible_duplicate",
    "supporting_evidence",
    name="duplicate_status",
    create_type=False,
)
report_status = ENUM(
    "pending",
    "verified",
    "in_progress",
    "resolved",
    "rejected",
    name="report_status",
    create_type=False,
)
review_status = ENUM("open", "resolved", name="review_status", create_type=False)

ENUM_TYPES = (
    "CREATE TYPE report_category AS ENUM ('refuse_dump', 'blocked_drain', 'flooding', 'pothole', 'pollution', 'broken_public_facility', 'sanitation', 'other')",
    "CREATE TYPE duplicate_status AS ENUM ('new', 'duplicate', 'possible_duplicate', 'supporting_evidence')",
    "CREATE TYPE report_status AS ENUM ('pending', 'verified', 'in_progress', 'resolved', 'rejected')",
    "CREATE TYPE review_status AS ENUM ('open', 'resolved')",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for statement in ENUM_TYPES:
        op.execute(
            f"""
            DO $$ BEGIN
                {statement};
            EXCEPTION
                WHEN duplicate_object THEN NULL;
            END $$;
            """
        )

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(160), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", report_category, nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column(
            "location_point", Geometry("POINT", srid=4326, spatial_index=True), nullable=False
        ),
        sa.Column("image_url", sa.Text(), nullable=False),
        sa.Column("image_sha256", sa.String(64), nullable=False, index=True),
        sa.Column("perceptual_hash", sa.String(32), nullable=True),
        sa.Column("image_embedding", Vector(512), nullable=True),
        sa.Column("detected_objects", sa.JSON(), nullable=True),
        sa.Column("cv_inferred_category", sa.String(64), nullable=True),
        sa.Column("cv_confidence_score", sa.Float(), nullable=True),
        sa.Column("duplicate_status", duplicate_status, nullable=False, server_default="new"),
        sa.Column(
            "duplicate_of_report_id",
            sa.Integer(),
            sa.ForeignKey("reports.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("status", report_status, nullable=False, server_default="pending"),
        sa.Column("source_ip", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_reports_category_created", "reports", ["category", "created_at"])
    op.create_index(
        "ix_reports_category_status_created", "reports", ["category", "status", "created_at"]
    )
    op.create_index("ix_reports_duplicate_status", "reports", ["duplicate_status"])
    op.create_index(
        "ix_reports_embedding_hnsw",
        "reports",
        ["image_embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"image_embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "duplicate_reviews",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "report_id",
            sa.Integer(),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "candidate_report_id",
            sa.Integer(),
            sa.ForeignKey("reports.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("status", review_status, nullable=False, server_default="open"),
        sa.Column("resolution", sa.String(64), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor_ip", sa.String(64), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column(
            "report_id",
            sa.Integer(),
            sa.ForeignKey("reports.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("duplicate_reviews")
    op.drop_index("ix_reports_embedding_hnsw", table_name="reports")
    op.drop_table("reports")
    op.execute("DROP TYPE IF EXISTS review_status")
    op.execute("DROP TYPE IF EXISTS report_status")
    op.execute("DROP TYPE IF EXISTS duplicate_status")
    op.execute("DROP TYPE IF EXISTS report_category")
