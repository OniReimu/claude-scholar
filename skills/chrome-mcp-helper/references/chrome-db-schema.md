# Chrome 数据库结构

## History 数据库

位置: `~/Library/Application Support/Google/Chrome/Default/History`

### urls 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| url | TEXT | URL 地址 |
| title | TEXT | 页面标题 |
| visit_count | INTEGER | 访问次数 |
| last_visit_time | INTEGER | 最后访问时间 (WebKit 格式) |
| typed_count | INTEGER | 手动输入次数 |

**时间转换公式:**
```sql
datetime(last_visit_time/1000000-11644473600, 'unixepoch', 'localtime')
```

### visits 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| url | INTEGER | 外键 -> urls.id |
| visit_time | INTEGER | 访问时间 |
| transition | INTEGER | 导航类型 |

## Bookmarks 文件

位置: `~/Library/Application Support/Google/Chrome/Default/Bookmarks`

格式: JSON

```json
{
  "roots": {
    "bookmark_bar": {
      "name": "书签栏",
      "type": "folder",
      "children": [...]
    },
    "other": {
      "name": "其他书签",
      "type": "folder",
      "children": [...]
    }
  }
}
```

## Cookies 数据库

位置: `~/Library/Application Support/Google/Chrome/Default/Cookies`

### cookies 表

| 字段 | 类型 | 说明 |
|------|------|------|
| host_key | TEXT | 域名 |
| name | TEXT | Cookie 名称 |
| value | TEXT | Cookie 值 |
| path | TEXT | 路径 |
| expires_utc | INTEGER | 过期时间 |
| secure | INTEGER | 是否 HTTPS only |
| httponly | INTEGER | 是否 HTTP only |

## 注意事项

1. **数据库锁定**: Chrome 运行时数据库可能被锁定，建议先复制
2. **时间格式**: Chrome 使用 WebKit 时间格式（1601 年 1 月 1 日以来的微秒数）
3. **备份**: 操作前建议备份原始数据库
