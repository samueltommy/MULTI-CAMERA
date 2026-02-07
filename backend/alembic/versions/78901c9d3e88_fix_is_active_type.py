"""fix is_active type

Revision ID: 78901c9d3e88
Revises: 23456b7c2d77
Create Date: 2026-02-07 14:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '78901c9d3e88'
down_revision = '23456b7c2d77'
branch_labels = None
depends_on = None


def upgrade():
    # Convert is_active from INTEGER to BOOLEAN
    # We use 'USING is_active::boolean' for PostgreSQL
    op.execute('ALTER TABLE calibrations ALTER COLUMN is_active TYPE BOOLEAN USING is_active::boolean')


def downgrade():
    op.execute('ALTER TABLE calibrations ALTER COLUMN is_active TYPE INTEGER USING is_active::integer')
