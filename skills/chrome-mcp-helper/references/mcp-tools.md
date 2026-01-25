# Chrome MCP 完整工具列表

## 浏览器管理工具

| 工具名称 | 功能描述 | 参数 |
|---------|---------|------|
| `get_windows_and_tabs` | 列出所有窗口和标签页 | 无 |
| `chrome_navigate` | 导航到指定 URL | url: string |
| `chrome_close_tabs` | 关闭指定标签页 | tabIds: number[] |
| `chrome_go_back_or_forward` | 浏览器前进后退 | direction: 'back' \| 'forward' |

## 内容分析工具

| 工具名称 | 功能描述 | 参数 |
|---------|---------|------|
| `chrome_get_web_content` | 获取页面文本/HTML | options: {format: 'text'\|'html'} |
| `search_tabs_content` | 语义搜索标签页内容 | query: string |
| `chrome_get_interactive_elements` | 查找可点击元素 | selector?: string |
| `chrome_console` | 获取控制台输出 | 无 |

## 交互操作工具

| 工具名称 | 功能描述 | 参数 |
|---------|---------|------|
| `chrome_click_element` | 点击页面元素 | selector: string |
| `chrome_fill_or_select` | 填写表单字段 | selector: string, value: string |
| `chrome_keyboard` | 模拟键盘输入 | key: string |

## 截图工具

| 工具名称 | 功能描述 | 参数 |
|---------|---------|------|
| `chrome_screenshot` | 高级截图 | options: {fullPage?: boolean, selector?: string} |

## 数据管理工具

| 工具名称 | 功能描述 | 参数 |
|---------|---------|------|
| `chrome_history` | 搜索浏览历史 | query: string, limit?: number |
| `chrome_bookmark_search` | 搜索书签 | query: string |
| `chrome_bookmark_add` | 添加书签 | url: string, title: string |
| `chrome_bookmark_delete` | 删除书签 | id: string |

## 网络监控工具

| 工具名称 | 功能描述 | 参数 |
|---------|---------|------|
| `chrome_network_capture_start` | 开始捕获网络请求 | 无 |
| `chrome_network_capture_stop` | 停止捕获并返回结果 | 无 |
| `chrome_network_request` | 发送 HTTP 请求 | url: string, options: RequestInit |
