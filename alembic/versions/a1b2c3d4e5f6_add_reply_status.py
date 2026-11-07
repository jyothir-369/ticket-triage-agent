"""Add tickets.reply_status column.

Revision ID: a1b2c3d4e5f6
Revises: None
Create Date: 2026-11-07
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tickets",
        sa.Column(
            "reply_status",
            sa.String(length=50),
            nullable=True,
            server_default="pending_approval",
            comment="Lifecycle status of the approved reply after the approve endpoint.",
        ),
    )


def downgrade() -> None:
    op.drop_column("tickets", "reply_status")
