#!/usr/bin/env python3
"""
快速搜索 Chrome 书签
从 Chrome 数据库直接读取，无需 MCP 调用
"""

import sqlite3
import os
from pathlib import Path

def get_chrome_bookmarks_path():
    """获取 Chrome 书签数据库路径"""
    home = Path.home()
    return home / "Library/Application Support/Google/Chrome/Default/Bookmarks"

def search_bookmarks_file(bookmarks_path: str, query: str):
    """从 Bookmarks JSON 文件搜索"""
    import json

    if not bookmarks_path.exists():
        return []

    with open(bookmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    query_lower = query.lower()

    def search_recursive(nodes, folder_path=""):
        for node in nodes:
            if node.get('type') == 'url':
                title = node.get('name', '')
                url = node.get('url', '')

                if query_lower in title.lower() or query_lower in url.lower():
                    results.append({
                        'title': title,
                        'url': url,
                        'folder': folder_path
                    })
            elif node.get('type') == 'folder':
                folder_name = node.get('name', '')
                search_recursive(node.get('children', []),
                               f"{folder_path}/{folder_name}" if folder_path else folder_name)

    # 搜索 bookmark bar 和 other
    if 'roots' in data:
        for root_key in ['bookmark_bar', 'other']:
            if root_key in data['roots']:
                root = data['roots'][root_key]
                search_recursive(root.get('children', []), root.get('name', ''))

    return results

def main():
    import sys

    if len(sys.argv) < 2:
        print("用法: search_bookmarks.py <搜索关键词>")
        return 1

    query = sys.argv[1]
    bookmarks_path = get_chrome_bookmarks_path()

    if not bookmarks_path.exists():
        print(f"错误: Chrome 书签文件未找到: {bookmarks_path}")
        return 1

    results = search_bookmarks_file(bookmarks_path, query)

    if not results:
        print(f"未找到包含 '{query}' 的书签")
        return 0

    print(f"找到 {len(results)} 个包含 '{query}' 的书签:")
    print("=" * 100)
    print(f"{'标题':<40} {'文件夹':<20} {'URL'}")
    print("=" * 100)

    for item in results:
        title = item['title']
        folder = item['folder']
        url = item['url']

        title = title[:37] + '...' if len(title) > 40 else title
        folder = folder[:17] + '...' if len(folder) > 20 else folder
        url = url[:50] + '...' if len(url) > 50 else url

        print(f"{title:<40} {folder:<20} {url}")

    return 0

if __name__ == "__main__":
    exit(main())
