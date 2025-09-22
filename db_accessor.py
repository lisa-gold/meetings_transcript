import sqlite3

class DBAccessor:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()

    def get_db_name(self):
        return self.db_name

    def get_connection(self):
        return self.connection

    def get_cursor(self):
        return self.cursor

    def execute_query(self, query: str):
        self.cursor.execute(query)
        self.connection.commit()

    def execute_query_with_params(self, query: str, params: tuple):
        self.cursor.execute(query, params)
        self.connection.commit()
