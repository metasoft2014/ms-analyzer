#-*- coding: utf-8 -*-
import urllib
import pymysql
import configparser
import pandas as pd
import os


class mariadb:
    def __init__(self, CONFIGDIR):
        config = configparser.ConfigParser()
        config.read(os.path.join(CONFIGDIR, 'config.ini'))
        self.host = config['DB']['HOST']
        self.port = int(config['DB']['PORT'])
        self.user = config['DB']['USERNAME']
        self.password = config['DB']['PASSWORD']
        self.database = config['DB']['DATABASE']
        self.charset = config['DB']['CHARSET']
        self.conn = self._get_connection()

    def _get_connection(self):
        # Open database connection
        return pymysql.connect(host=self.host, port=self.port, user=self.user, passwd=self.password, db=self.database, charset=self.charset, autocommit=True)

    def close_conneciton(self):
        self.conn.close()

    def __del__(self):
        self.close_conneciton()

    def get_content_from_db(self, table_name):
        sql = 'SELECT * FROM {0}'.format(table_name)

        try:
            with self.conn.cursor() as cursor:
                df = pd.read_sql(sql, con=self.conn)

        finally:
            cursor.close();

        return df

if __name__ == "__main__":
    db = mariadb()
    df = db.get_content_from_db('ms_collected_data')
    print(df.head())