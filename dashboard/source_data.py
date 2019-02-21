import json
import pandas as pd
import numpy as np
from pymongo import MongoClient

class Mongo:
    def __init__(self, client):
        self.cleint = client
        self.datas = None

    def fetch(self, db, collection, key=None):
        '''
        return list of dict
        '''
        database = self.cleint[db]
        col = database[collection]
        if col.count() < 8760 :
            cursor = col.find()
        else:
            cursor = col.find().skip(col.count() - 8760)
            self.datas = [ docu for docu in cursor ]
        return self
        
    def toDf(self):
        return pd.DataFrame(
            self.datas
        )
    
    def _get_value(self, data):
        print(data)
        if isinstance(data, str):
            return data
        else: 
            return data[0]
        
# # test
ip = "localhost"
port = "27017"
address = "mongodb://{0}:{1}/".format(ip, port)
client = MongoClient(address)
mongo = Mongo(client)
res = mongo.fetch('aiCar-emission-prediction', 'carA11548286102687')
data = res.toDf()
# if data.shape[0] > 50:
    # data = data.iloc[range(data.shape[0]-50, data.shape[0]), :]

# data = data
# data = np.reshape(data.values, -1, 1).tolist()


def get_connection():
    from pymongo import MongoClient
    ip = "112.169.120.25"
    ip = "172.30.1.2"
    port = "27017"
    address = "mongodb://{0}:{1}/".format(ip, port)
    return MongoClient(address)

def query_data(client, db, collection, query_field):
    def get_value(data):
        if isinstance(data, str):
            return data
        else:
            return data[0]
    
    database = client[db]
    col = database[collection]
    cursor = col.find({}, {query_field:1})

    return pd.DataFrame(
        { 
            query_field :[ get_value(docu[query_field]) for docu in cursor ] 
        }
    )

def provide_data(conn, num_count=100):
    nox = query_data(conn, "emission", "prediction", "NOx").cumsum()
    time = query_data(conn, "emission", "inputs", "TIME").iloc[2:,:].reset_index()
    df = pd.merge(nox, time, left_index=True, right_index=True, how='inner')
    total_num = nox.shape[0]
    return df.iloc[range(total_num-num_count, total_num),:]\
             .reset_index()\
             .drop(['level_0', 'index'], axis=1)


class KafKaHLConsumer:
    def __init__(self, config, topics, partitions=[0], timeout=1.0):
        '''
        @params 
        config : shoud contain at least :
            - bootstrap.servers: 'mybroker'
            - group.id : 'mygroup'

        topics : list of topics
        partitions : lsit of partitions 
        '''
        from confluent_kafka import Consumer
        self.consumer = Consumer(config)
        self.consumer.subscribe(topics)
        self.timeout = timeout
        print('kafka consumer initiated')


    def fetch_one(self):
        while True:
            msg = self._fetch_one()
            if msg:
                return msg 


    def _fetch_one(self):
        msg = self.consumer.poll(self.timeout)
        if msg is None:
            return None
        elif msg.error():
            return None
        value = json.loads(msg.value().decode('utf-8'))
        return value
        
    def close(self):
        self.consumer.close()
    