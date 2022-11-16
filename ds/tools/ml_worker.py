import traceback
import pandas as pd
from sqlalchemy import create_engine
from time import sleep
import uuid



class PostgresMeta:
    
    engine = None

    def __init__(self, conn):
        self.conn = conn
        if PostgresMeta.engine is None:
            PostgresMeta.engine = create_engine(self.conn, 
                                    isolation_level = "REPEATABLE READ")
        self.engine = PostgresMeta.engine
        
        
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
        self.handlers = set()
        
    def attach(self):
        if not self.pm.has_col(self.table, self.track_col):
            self.pm.add_col(self.table, self.track_col, "TIMESTAMPTZ")        
    
    def detach(self):
        self.handlers = set()
    
    def add_handler(self, handler):
        self.handlers.add(handler)
    
    def reset(self):
        self.handlers = set()
        if self.pm.has_col(self.table, self.track_col):
            self.pm.del_col(self.table, self.track_col)
        
    def _notify(self, rec):
        for h in self.handlers:
            
            try:
                h(rec)
            except:
                traceback.print_exc()
                return False
            
        return True
        
    def check(self):
        new_recs = pd.read_sql(f"SELECT * FROM {self.table} WHERE {self.track_col} != {self.ts_col} OR {self.track_col} is NULL", con=self.conn)
        for _, rec in new_recs.iterrows():
            if self._notify(rec): 
                self.mark(rec[self.uuid_col])
    
    def mark(self, uuid):
        self.pm.query(f"UPDATE {self.table} SET {self.track_col} = {self.ts_col} where {self.uuid_col} = '{uuid}'")
        
    
    
    
    
from utils import word2vec
import numpy as np
import torch
import hnswlib    



def combine(vecs, norm='l2'):
    rs = torch.zeros(768)
    n = 0
    for v, w in vecs:
        if not v is None and v == v:
            rs += word2vec(v) * w
            n  += w
    #rs = rs / n if n > 0 else rs
    rs = rs / torch.sum(rs * rs)
    return rs

# def get_usr_liked(uid, con):
#     likes = pd.read_sql_query("select * from likes where author_uuid='%s'" % uid , con=con)
#     return likes

def get_usr_liked(uid, con):
    likes = pd.read_sql_query(
        "select * from ideas ii join likes ll on ii.uuid=ll.idea_uuid where ll.author_uuid='%s' order by ll.created_at desc" % uid , con=con)
    return likes.drop_duplicates()

def get_liked(idea_uid, con):
    likes = pd.read_sql_query(
        "select ll.author_uuid, name, description, is_positive, category_uuid "\
        "from ideas ii join likes ll on ii.uuid=ll.idea_uuid where ll.idea_uuid='%s' order by ll.created_at desc" % idea_uid , con=con)
    return likes.drop_duplicates()


def get_usr_by_uid(uid, con):
    user = pd.read_sql_query("select * from users where uuid='%s'" % uid , con=con)
    return user.iloc[0]


def get_idea_by_uid(iid, con):
    idea = pd.read_sql_query("select * from ideas where uuid='%s'" % iid , con=con)
    return idea.iloc[0]


class FeedRefresher:
    
    def __init__(self, conn, table, user_col="user_uuid", item_col="idea_uuid", score_col="level"):
        self.conn = conn
        self.table = table
        self.pm = PostgresMeta(conn)
        self.item_col = item_col
        self.user_col = user_col
        self.score_col = score_col
        
    def clear_for_user(self, user_uuid):
        self.pm.query(f"DELETE FROM {self.table} where {self.user_col} = '{user_uuid}'")
    
    
    def put_new_item(self, user_uuid, item_uuid, level):
        self.pm.query(
            f"INSERT into {self.table} ({self.user_col}, {self.item_col},"\
            f" {self.score_col}, created_at, updated_at) values ('{user_uuid}', '{item_uuid}', {level}, NOW(), NOW())") 
        
    def put_new_like(self, user_uuid, item_uuid, is_positive):
        uuid4s = str(uuid.uuid4())
        self.pm.query(
            f"INSERT into likes (uuid, author_uuid, idea_uuid,"\
            f" is_positive, created_at, updated_at) values ('{uuid4s}', '{user_uuid}', '{item_uuid}', {is_positive}, NOW(), NOW())")  
        
        
    def put_new_team_request(self, user_uuid, item_uuid, level):
        uuid4s = str(uuid.uuid4())
        self.pm.query(
            f"INSERT into team_requests_advices (uuid, user_uuid, idea_uuid,"\
            f" level, created_at, updated_at) values ('{uuid4s}', '{user_uuid}', '{item_uuid}', {level}, NOW(), NOW())")         
            
    

class ML_Ideas_Feeder_Baseline():
    
    
    def __init__(self, conn, n_max_items=100000):
        self.conn = conn
        # av. spaces: l2, cosine or ip
        self.idx = hnswlib.Index(space = 'cosine', dim = 768) 
        self.idx.init_index(max_elements = n_max_items, ef_construction = 300, M = 32)
        self.uuids = {}
        self.feeder = FeedRefresher(conn, 'likes_advices', 'user_uuid', 'idea_uuid')
        self.categories = pd.read_sql("select uuid, name from idea_categories", con=conn)
        self.categories.set_index("uuid", inplace=True)
        
    def category_to_text(self, cat_id):
        if not cat_id is None:
            return self.categories.loc[cat_id, "name"]
        
    def __update_ideas_index(self, idea):
        vec = combine([(idea["name"], 1), (idea.description,1), (self.category_to_text(idea.category_uuid),5)])
        self.uuids[self.idx.element_count] = idea["uuid"]
        self.idx.add_items(vec)
    
    # def __handle_user_like(self, like):
    #     # get_usr_by_uid(like.author_uuid)
    #     if like.is_positive:
    #         author_uuid = like.author_uuid
    #         idea = get_idea_by_uid(like.idea_uuid, self.conn)
    #         query = combine(idea["name"], idea.description)
    #         labels, distances = self.idx.knn_query(query)
    #         self.feeder.clear_for_user(author_uuid)
    #         new = [self.uuids[s] for s in labels.reshape(-1)]
    #         distances  = distances.reshape(-1)
    #         print(new)
    #         raise 1
    #         for idea_uuid, d in zip(new, distances):
    #             print("updating: ", idea_uuid)
    #             self.feeder.put_new_item(author_uuid, idea_uuid, 1 - d)
    
    def __handle_user_like(self, like, top_k=5):
        liked_ideas = get_usr_liked(like.author_uuid, con=self.conn)
        candidates = []
        for _, idea  in liked_ideas.head(top_k).iterrows():
            query = combine([(idea["name"], 1), (idea.description,1), (self.category_to_text(idea.category_uuid),5)])
            labels, distances = self.idx.knn_query(query, 10)
            labels = [self.uuids[s] for s in labels.reshape(-1)]
            sign = 1 if idea.is_positive else -1
            similarity = (1. - distances.reshape(-1)) * sign
            candidates += [ pd.DataFrame({"idea_uuid":labels, "level":similarity})]

        candidates = pd.concat(candidates)
        candidates = candidates.groupby("idea_uuid")["level"].mean()
        # self.feeder.clear_for_user(like.author_uuid)
        for idea_uuid, level in candidates.iteritems():
            self.feeder.put_new_item(like.author_uuid, idea_uuid, level * 100)
        
        
    def __notify_on_new_ideas(self, idea, top_k=50, top_users=5):
        query = combine([(idea["name"], 1), (idea.description,1), (self.category_to_text(idea.category_uuid),5)])
        labels, distances = self.idx.knn_query(query, top_k)
        labels = [self.uuids[s] for s in labels.reshape(-1)]
        
        variants = []
        for label in labels:
            ideas = get_liked(label, con=self.conn)
            variants += [ideas]
            
        users = pd.concat(variants)
        users["is_positive"] = users["is_positive"].fillna(0.).astype(int)
        
        if users.empty: return
        
        levels = users.groupby("author_uuid")["is_positive"].mean().sort_values(ascending=False)
        levels = levels.head(5)
        for uuid, level in levels.iteritems():
            print("notifying: ", idea.uuid, uuid, level * 100)
            self.feeder.put_new_team_request(uuid, idea.uuid, level)
            
        
    def start(self):
        print("starting")
        
        self.ideas_tracker = DBTracker(self.conn, "ideas", "uuid", "updated_at")
        self.ideas_tracker.reset()
        self.ideas_tracker.add_handler(self.__update_ideas_index)
        self.ideas_tracker.attach()
        self.ideas_tracker.check()
        print("loaded")
        self.ideas_tracker.add_handler(self.__notify_on_new_ideas)
        
        
        self.likes_tracker = DBTracker(self.conn, "likes", "uuid", "updated_at")
        self.likes_tracker.attach()
        self.likes_tracker.add_handler(self.__handle_user_like)
        
        print("working")
        while True: 
            sleep(2)
            self.ideas_tracker.check()
            self.likes_tracker.check()
        
        
        
        
        