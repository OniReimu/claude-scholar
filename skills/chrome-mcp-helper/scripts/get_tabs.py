#!/usr/bin/env python3
"""
快速获取 Chrome 当前标签页
从 Chrome 数据库直接读取，无需 MCP 调用
"""

import sqlite3
import shutil
import os
from pathlib import Path
from datetime import datetime

def get_chrome_db_path():
    """获取 Chrome 历史数据库路径"""
    home = Path.home()
    return home / "Library/Application Support/Google/Chrome/Default/History"

def get_tabs_from_db(db_path: str, limit: int = 20):
    """从数据库获取最近的标签页/历史记录"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
    SELECT
        datetime(last_visit_time/1000000-11644473600, 'unixepoch', 'localtime') as visit_time,
        title,
        url
    FROM urls
    ORDER BY last_visit_time DESC
    LIMIT ?
    """

    cursor.execute(query, (limit,))
    results = cursor.fetchall()
    conn.close()

    return results

def main():
    db_path = get_chrome_db_path()

    if not db_path.exists():
        print(f"错误: Chrome 历史数据库未找到: {db_path}")
        return 1

    # 复制数据库以避免锁定
    temp_db = "/tmp/chrome_history_tabs.db"
    try:
        shutil.copy(db_path, temp_db)
    except Exception as e:
        print(f"错误: 无法复制数据库: {e}")
        return 1

    try:
        results = get_tabs_from_db(temp_db, limit=20)

        print("=" * 80)
        print(f"{'访问时间':<20} {'标题':<40} {'URL'}")
        print("=" * 80)

        for row in results:
            time = row['visit_time']
            title = row['title'] or '(无标题)'
            url = row['url']

            # 截断过长的标题和 URL
            title = title[:37] + '...' if len(title) > 40 else title
            url = url[:50] + '...' if len(url) > 50 else url

            print(f"{time:<20} {title:<40} {url}")

    finally:
        os.unlink(temp_db)

    return 0

if __name__ == "__main__":
    exit(main())
