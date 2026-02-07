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
    # PostgreSQL doesn't allow direct integer -> boolean cast without explicit USING
    # We check the current type first to avoid redundant errors
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = inspector.get_columns('calibrations')
    is_active_col = next((c for c in columns if c['name'] == 'is_active'), None)
    
    if is_active_col and not isinstance(is_active_col['type'], sa.Boolean):
        op.execute('ALTER TABLE calibrations ALTER COLUMN is_active TYPE BOOLEAN USING (CASE WHEN is_active=1 THEN TRUE ELSE FALSE END)')


def downgrade():
    op.execute('ALTER TABLE calibrations ALTER COLUMN is_active TYPE INTEGER USING (CASE WHEN is_active=TRUE THEN 1 ELSE 0 END)')
