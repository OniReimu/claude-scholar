#!/usr/bin/env python3
"""
快速搜索 Chrome 浏览历史
从 Chrome 数据库直接读取，无需 MCP 调用
"""

import sqlite3
import shutil
import os
import sys
from pathlib import Path
from datetime import datetime

def get_chrome_db_path():
    """获取 Chrome 历史数据库路径"""
    home = Path.home()
    return home / "Library/Application Support/Google/Chrome/Default/History"

def search_history(db_path: str, query: str, limit: int = 50):
    """搜索浏览历史"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    search_pattern = f"%{query}%"
    sql_query = """
    SELECT
        datetime(last_visit_time/1000000-11644473600, 'unixepoch', 'localtime') as visit_time,
        title,
        url,
        visit_count
    FROM urls
    WHERE title LIKE ? OR url LIKE ?
    ORDER BY last_visit_time DESC
    LIMIT ?
    """

    cursor.execute(sql_query, (search_pattern, search_pattern, limit))
    results = cursor.fetchall()
    conn.close()

    return results

def main():
    if len(sys.argv) < 2:
        print("用法: search_history.py <搜索关键词> [结果数量]")
        return 1

    query = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    db_path = get_chrome_db_path()

    if not db_path.exists():
        print(f"错误: Chrome 历史数据库未找到: {db_path}")
        return 1

    # 复制数据库以避免锁定
    temp_db = "/tmp/chrome_history_search.db"
    try:
        shutil.copy(db_path, temp_db)
    except Exception as e:
        print(f"错误: 无法复制数据库: {e}")
        return 1

    try:
        results = search_history(temp_db, query, limit)

        if not results:
            print(f"未找到包含 '{query}' 的浏览记录")
            return 0

        print(f"找到 {len(results)} 条包含 '{query}' 的浏览记录:")
        print("=" * 100)
        print(f"{'访问时间':<20} {'标题':<40} {'URL'}")
        print("=" * 100)

        for row in results:
            time = row['visit_time']
            title = row['title'] or '(无标题)'
            url = row['url']

            # 高亮匹配
            title = title[:37] + '...' if len(title) > 40 else title
            url = url[:50] + '...' if len(url) > 50 else url

            print(f"{time:<20} {title:<40} {url}")

    finally:
        os.unlink(temp_db)

    return 0

if __name__ == "__main__":
    exit(main())
