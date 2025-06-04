import pyodbc
from sqlalchemy import create_engine, text

# Database configuration
server = 'ANTOINEPC\\MSSQLSERVER01' # Update with your server name
database = 'racing_data'

def set_database(db_name):
    global database
    database = db_name

driver = '{ODBC Driver 17 for SQL Server}'

def ConnectToDatabase():
    """
    Establishes a connection to the database using pyodbc.
    Returns:
        conn: A pyodbc connection object.
    """
    try:
        conn = pyodbc.connect(
            f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
        )
        print("Connection successful!")
        return conn
    except Exception as e:
        print(f"Error: Unable to connect to the database. {e}")
        return None


def ReturnConnectionString():
    """
    Returns a SQLAlchemy-compatible connection string.
    Returns:
        str: Connection string for SQLAlchemy.
    """
    return f"mssql+pyodbc://@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"





def get_last_inserted_id(table_name, log=False):
    """
    Retrieves the last inserted ID from a specified table.
    Args:
        table_name (str): The name of the table to query.
        log (bool): Enable or disable logging of the retrieved ID.
    Returns:
        int or None: The last inserted ID.
    """
    try:
        # Validate table name (basic check)
        if not table_name.isidentifier():
            raise ValueError("Invalid table name provided.")

        # Build the SQL query
        query = text(f"SELECT IDENT_CURRENT('{table_name}') AS LastInsertedID")

        # Create the SQLAlchemy engine
        connection_string = ReturnConnectionString()
        engine = create_engine(connection_string)

        # Execute the query and fetch the last inserted ID
        with engine.connect() as conn:
            result = conn.execute(query)
            last_id = result.scalar()  # Fetch the first column value

        if log:
            print(f"Last inserted ID in {table_name}: {last_id}")
        return last_id
    except Exception as e:
        print(f"Error while retrieving last inserted ID: {e}")
        return None


def insert_dataframe_to_db(df, table_name, getlastid=False, chunk_size=1000, log=False):
    """
    Inserts a DataFrame into a database table using pandas.to_sql.
    Args:
        df (DataFrame): The DataFrame to insert.
        table_name (str): The name of the target table.
        getlastid (bool): Whether to return the last inserted ID.
        chunk_size (int): Number of rows per batch for insertion.
        log (bool): Enable or disable logging of success messages.
    Returns:
        int or None: The last inserted ID if getlastid is True, otherwise None.
    """
    try:
        connection_string = ReturnConnectionString()
        engine = create_engine(connection_string)
        
        # Insert with batching
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False, chunksize=chunk_size)
        if log:
            print(f"Data successfully inserted into table '{table_name}'.")
        
        if getlastid:
            return get_last_inserted_id(table_name)
        else:
            return None
    except Exception as e:
        print(f"Error during insertion: {e}")
        return 1

def execute_query(query, params=None):
    """
    Executes a raw SQL query.
    Args:
        query (str): The SQL query to execute.
        params (dict): Parameters to bind to the query.
    Returns:
        result: The result of the query execution.
    """
    try:
        connection_string = ReturnConnectionString()
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            if query.strip().lower().startswith("select"):
                # Fetch all rows for SELECT queries
                return result.fetchall()
            else:
                # Commit for non-SELECT queries (INSERT, UPDATE, DELETE)
                conn.commit()
                return True
    except Exception as e:
        print(f"Error executing query: {e}")
        return None




