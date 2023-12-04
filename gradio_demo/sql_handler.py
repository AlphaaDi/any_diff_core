import sqlite3
from datetime import datetime
import uuid
import json


def parse_task(task):
    return {
        'task_id': task[0],
        'timestamp': task[1],
        'objects': json.loads(task[2]),
        'original_video_path': task[3],
        'config_path': task[4],
        'status': task[5],
        'file_path': task[6],
    }


class TaskDatabase:
    def __init__(self, db_path):
        """
        Initialize the database connection and create the table if it doesn't exist.
        :param db_path: Path to the SQLite database file.
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()


    def get_query_result(self, query):
        with self.conn as conn: 
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return rows


    def create_table(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS tasks
                                 (task_id TEXT PRIMARY KEY,
                                  timestamp TEXT,
                                  objects_json TEXT,
                                  original_video_path TEXT,
                                  config_path TEXT,
                                  status TEXT,
                                  file_path TEXT)''')

    def insert_task(self, objects, original_video_path, config_path, task_id, status='wait', file_path=''):
        """
        Insert a new task into the database.
        :param objects: A dictionary of objects with keys like 'Object_1'.
        :param status: The status of the task ('wait', 'in progress', 'ready').
        :param file_path: The file path to the result.
        :return: The unique task ID.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        current_timestamp = datetime.now().isoformat()
        objects_json = json.dumps(objects)


        with self.conn:
            self.conn.execute("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (task_id, current_timestamp, objects_json, original_video_path, config_path, status, file_path))

        return task_id

    def update_task_status(self, task_id, file_path='', status='ready'):
        """
        Update the status and file path of a task.
        :param task_id: The unique task ID.
        :param status: The new status of the task.
        :param file_path: The new file path for the task result.
        """
        with self.conn:
            self.conn.execute("UPDATE tasks SET status=?, file_path=? WHERE task_id=?",
                              (status, file_path, task_id))


    def retrieve_oldest_wait_task(self, status='wait'):
        """
        Retrieve the oldest task from the database.
        :return: A dictionary containing task details or None if not found.
        """
        with self.conn:
            task = self.conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY timestamp ASC LIMIT 1", (status,)
            ).fetchone()

        if task:
            return parse_task(task)
        else:
            return None


    def retrieve_all_task(self, status='*'):
        if status == '*':
            with self.conn:
                tasks = self.conn.execute(
                    "SELECT * FROM tasks ORDER BY timestamp ASC"
                ).fetchall()
        else:
            with self.conn:
                tasks = self.conn.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY timestamp ASC", (status,)
                ).fetchall()

        if tasks:
            return [parse_task(task) for task in tasks]
        else:
            return None


    def close(self):
        self.conn.close()


    def remove_task(self, task_id):
        with self.conn:
            self.conn.execute("DELETE FROM tasks WHERE task_id=?", (task_id,))


    def remove_all_task(self):
        with self.conn:
            self.conn.execute("DELETE FROM tasks WHERE 1=1")