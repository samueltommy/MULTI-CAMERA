import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def setup_database():
    # Load .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)

    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("Error: DATABASE_URL not found in .env")
        return

    # Parse URL manually for psycopg2 (simplified)
    # Expected format: postgresql://user:pass@host:port/dbname
    try:
        # Strip 'postgresql://'
        clean_url = db_url.replace('postgresql://', '')
        # Split user:pass and host:port/dbname
        auth, rest = clean_url.split('@')
        user, password = auth.split(':')
        # Split host:port and dbname
        connection_info, db_name = rest.split('/')
        
        # Split host and port
        if ':' in connection_info:
            host, port = connection_info.split(':')
        else:
            host = connection_info
            port = '5432'
    except Exception as e:
        print(f"Error parsing DATABASE_URL: {e}")
        return

    print(f"Connecting to PostgreSQL at {host} to check for database '{db_name}'...")

    try:
        # Connect to default 'postgres' database to create the new one
        conn = psycopg2.connect(
            dbname='postgres',
            user=user,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()

        if not exists:
            print(f"Database '{db_name}' does not exist. Creating it...")
            cur.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")

        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    setup_database()
