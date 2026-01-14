import json
import os
import re  # 新增：用于正则匹配
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# --- 配置路径 ---
PARENT_FILE = "parent_comments.json"  # 上下文源文件
USER_POSTS_FILE = "user_posts_sorted.json"  # 目标用户回复文件
OUTPUT_FILE = "train.jsonl"  # 输出文件

# --- 配置参数 ---
MIN_HISTORY = 5  # 用户的历史评论少于这个数量则跳过该用户（冷启动问题）
MAX_HISTORY_SAVE = 50  # 存入JSONL时最多保留多少条历史（节省空间，具体由Dataset类再截断）


def parse_time(time_str):
    """
    解析时间字符串，格式样例: "23/3/2023 15:59:40"
    """
    try:
        # 尝试标准格式 %d/%m/%Y %H:%M:%S
        return datetime.strptime(time_str, "%d/%m/%Y %H:%M:%S")
    except ValueError:
        try:
            # 容错处理：有时日期只有一位数可能导致问题，但标准库通常能处理
            # 这里可以添加其他格式的容错
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except:
            return None


def load_json_list(filepath):
    """读取JSON列表文件"""
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filepath}")
            return []


def contains_url(text):
    """
    检查文本是否包含URL链接
    """
    if not text:
        return False
    # 匹配 http:// 或 https:// 开头的链接
    url_pattern = re.compile(r'https?://\S+')
    return bool(url_pattern.search(text))


def build_context_chain(current_post, post_db):
    """
    递归查找父级评论，构建对话上下文
    返回列表: [最顶层帖子内容, ..., 直接父级评论内容]
    """
    chain = []
    curr = current_post
    depth = 0
    max_depth = 5  # 防止死循环或过深

    # 只要存在 parent_id 且能在数据库中找到
    while curr.get('parent_id') and curr['parent_id'] in post_db and depth < max_depth:
        parent = post_db[curr['parent_id']]
        # 将父级内容插入到列表最前面
        content = parent.get('content', '').strip()
        if content:
            chain.insert(0, content)  # 越父级越靠前

        curr = parent
        depth += 1

    return chain


def main():
    # 1. 加载数据
    parents = load_json_list(PARENT_FILE)
    users = load_json_list(USER_POSTS_FILE)

    # 2. 构建全局帖子索引 (用于查找 Context)
    # 我们把 parents 和 users 都放进去，因为子评论的父级可能是另一个用户的子评论
    post_db = {}
    print("Building post index...")
    for p in parents:
        post_db[p['post_id']] = p
    for u in users:
        post_db[u['post_id']] = u

    # 3. 按 UID 分组并排序
    print("Grouping and sorting user history...")
    user_groups = defaultdict(list)
    for u in users:
        # 解析时间用于排序
        dt = parse_time(u['pubtime'])
        if dt:
            u['_dt'] = dt  # 临时存一下datetime对象
            user_groups[u['uid']].append(u)

    # 4. 生成训练样本
    print("Generating training samples...")
    valid_samples_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for uid, raw_posts in tqdm(user_groups.items(), desc="Processing Users"):
            # A. 按时间排序：旧 -> 新
            raw_posts.sort(key=lambda x: x['_dt'])

            # B. 数据清洗：剔除包含URL的帖子
            posts = []
            for p in raw_posts:
                content = p.get('content', '').strip()
                # 如果有内容且不包含URL，则保留
                if content and not contains_url(content):
                    p['content'] = content  # 更新去除空白后的内容
                    posts.append(p)

            # C. 如果清洗后该用户数据太少，无法构建有效的 History，跳过
            if len(posts) < MIN_HISTORY + 1:
                continue

            # 提取纯文本内容列表（按时间序）
            # 格式化一下：加入时间信息可能有助于模型理解，或者只用content
            # 这里简单起见只用 content，你可以改成 f"{p['pubtime']} {p['content']}"
            all_contents = [p['content'] for p in posts]

            # 滑动窗口生成样本
            # 从第 MIN_HISTORY 个帖子开始作为 Target，之前的作为 History
            for i in range(MIN_HISTORY, len(posts)):
                target_post = posts[i]

                # a. 构建 History
                # 取当前 Target 之前的最多 MAX_HISTORY_SAVE 条
                start_idx = max(0, i - MAX_HISTORY_SAVE)
                history_segment = all_contents[start_idx:i]

                # b. 构建 Context (递归查找父级)
                context_chain = build_context_chain(target_post, post_db)

                # 如果找不到上下文（比如是一条孤立的回复，且parent不在库里），
                # 策略：可以选择跳过，或者仅保留 content。
                # 既然是回复训练，建议必须有上下文。
                if not context_chain:
                    continue

                # 将 Context 列表拼接成字符串
                # 格式:
                # [Post] 楼主内容
                # [Comment] 楼中楼内容
                # ...
                context_str = "\n---\n".join(context_chain)

                # c. 构建 Target
                target_str = target_post['content']

                # d. 写入
                sample = {
                    "uid": uid,
                    "history": history_segment,
                    "context": context_str,
                    "target": target_str,
                    "meta": {
                        "pubtime": target_post['pubtime'],
                        "post_id": target_post['post_id']
                    }
                }

                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                valid_samples_count += 1

    print(f"Done! Generated {valid_samples_count} samples in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()