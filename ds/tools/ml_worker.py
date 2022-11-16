import traceback
import pandas as pd
from sqlalchemy import create_engine

class PostgresMeta:

    def __init__(self, conn):
        self.conn = conn
        self.engine = create_engine(self.conn, 
                                    isolation_level = "REPEATABLE READ")
        
    def get_table_cols(self, table):
        result = pd.read_sql(f"SELECT * FROM information_schema.columns WHERE table_name='{table}'", con=self.conn)
        return result
    
    def has_col(self, table, col):
        cur = self.engine.execute(f"SELECT 1 FROM information_schema.columns WHERE table_name='{table}' AND column_name='{col}'") 
        return cur.rowcount == 1
    
    def add_col(self, table, col, coltype):
        cur = self.engine.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}") 
        return cur
    
    def del_col(self, table, col):
        cur = self.engine.execute(f"ALTER TABLE {table} DROP COLUMN {col}")    
        return cur
        
    def query(self, query):
        cur = self.engine.execute(query)
        return cur
    

class DBTracker:
    
    def __init__(self, conn, table, uuid_col, ts_col, prefix='dbt_'):
        self.conn = conn
        self.table = table
        self.pm = PostgresMeta(conn)
        self.uuid_col = uuid_col
        self.ts_col = ts_col
        self.track_col = prefix + ts_col
        self.handlers = []
        
    def attach(self):
        if not self.pm.has_col(self.table, self.track_col):
            self.pm.add_col(self.table, self.track_col, "TIMESTAMPTZ")        
    
    def detach(self):
        self.handlers = []
    
    def add_handler(self, handler):
        self.handlers.add(handler)
    
    def reset(self):
        self.handlers = []
        if self.pm.has_col(self.table, self.track_col):
            self.pm.del_col(self.table, self.track_col)
        
    def _notify(self, rec):
        for h in self.handlers
            try:
                h(rec)
            except:
                traceback.print_exc()
        return True
        
    def check(self):
        new_recs = pd.read_sql(f"SELECT * FROM {self.table} WHERE {self.track_col} != {self.ts_col} OR {self.track_col} is NULL", con=self.conn)
        for _, rec in new_recs.iterrows():
            if self._notify(rec): 
                self.mark(rec[self.uuid_col])
    
    def mark(self, uuid):
        self.pm.query(f"UPDATE {self.table} SET {self.track_col} = NOW() where {self.uuid_col} = '{uuid}'")
        
    