import yaml
import json
import praw

api_file_path = 'api_keys.json'

with open(api_file_path, 'r') as file:
    api_data = json.load(file)  

reddit = praw.Reddit(client_id=api_data['praw']['client_id'],
                    client_secret=api_data['praw']['client_secret'],
                    user_agent=api_data['praw']['user_agent'],
                    username=api_data['praw']['username'],
                    check_for_async=False) 

cfg_file_path = 'casaai_config.yaml'

with open(cfg_file_path, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)  
    
gemini_generated_keywords = cfg.get('gemini_generated_keywords', [])
user_provided_keywords = cfg.get('user_provided_keywords', [])
broad_keywords = cfg.get('broad_keywords', [])

def search_subreddits(keyword):
    subreddits = []
    for subreddit in reddit.subreddits.search_by_name(keyword, exact=False):
        subreddits.append(subreddit.display_name)
    return subreddits

def search_posts(subreddit_name, keyword):
    posts = []
    post_ids = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.search(keyword, limit=10):
        post_data = {
            'title': post.title,
            "comment_id": post.id,
            'url': post.url,
            'score': post.score,
            'num_comments': post.num_comments,
            'Post_views': post.view_count,
            'upvote_ratio': post.upvote_ratio,
            'author': str(post.author),
            'created_utc': post.created_utc,
            'image_urls': [],
            'comments': []
        }

        # Check for images in the post
        if hasattr(post, 'url') and (post.url.endswith('.jpg') or post.url.endswith('.png')):
            post_data['image_urls'].append(post.url)
        elif hasattr(post, 'media_metadata'):
            for item_id in post.media_metadata:
                media_item = post.media_metadata[item_id]
                if 'm' in media_item and 'image' in media_item['m']:
                    url = media_item.get('s', {}).get('u', None)
                    if url:
                        post_data['image_urls'].append(url)

        # Fetching comments
        post.comments.replace_more(limit=0)
        for comment in post.comments.list():
            comment_data = {
                "comment_id": comment.id,
                "parent_id": comment.parent_id.split('_')[1],
                "text": comment.body,
                "author": str(comment.author),
                "score": comment.score,
                "created_utc": comment.created_utc,
                "image_url": ""
            }

            # Check for images in comments if applicable
            if hasattr(comment, 'body_html') and 'img src="' in comment.body_html:
                start_index = comment.body_html.find('img src="') + len('img src="')
                end_index = comment.body_html.find('"', start_index)
                comment_data['image_url'] = comment.body_html[start_index:end_index]

            post_data['comments'].append(comment_data)

        posts.append(post_data)
        post_ids.append(post.id)
    
    return posts, post_ids

def fetch_reddit():
    
    found_subreddits = [search_subreddits(keyword) for keyword in broad_keywords]

    all_subreddits = []
    for each in found_subreddits:
        all_subreddits.extend(each)
    all_subreddits = set(all_subreddits)

    reddit_posts = {}
    reddit_post_ids = []

    for subreddit_name in all_subreddits:
        reddit_posts[subreddit_name] = {}
        for keyword in gemini_generated_keywords:
            posts, post_ids = search_posts(subreddit_name, keyword)    
            reddit_posts[subreddit_name][keyword] = posts
            reddit_post_ids.extend(post_ids)
    
    return reddit_posts, reddit_post_ids 

def condense_data(reddit_posts, reddit_post_ids):
    condensed_data = []
    unique_post_ids = set()
    unique_comment_ids = set()
    dup_post_cnt = 0
    unq_post_cnt = 0
    dup_comm_cnt = 0
    unq_comm_cnt = 0
    tot_post_cnt = 0
    tot_comm_cnt = 0
    cond_data_cnt = 0
    
    unq_reddit_post_ids = set()
    for p_id in reddit_post_ids:
        unq_reddit_post_ids.add(p_id)
    
    for subreddit_name, keywords_posts in reddit_posts.items():
        for keyword, posts in keywords_posts.items():
            for post in posts:
                tot_post_cnt += 1
                if post['comment_id'] in unique_post_ids:
                    dup_post_cnt += 1
                else:
                    unq_post_cnt += 1
                    post_comments = [{
                        'comment_id': post['comment_id'],
                        'parent_id': None,  # The main post has no parent
                        'text': post['title'],
                        'author': post['author'],
                        'score': post['score'],
                    }]
                    unique_post_ids.add(post['comment_id'])
                    for comment in post['comments']:
                        tot_comm_cnt += 1
                        if comment['comment_id'] in unique_comment_ids:
                            dup_comm_cnt += 1                        
                        else:
                            unq_comm_cnt += 1
                            comment_data = {
                                'comment_id': comment['comment_id'],
                                'parent_id': comment['parent_id'],
                                'text': comment['text'],
                                'author': comment['author'],
                                'score': comment['score'],
                            }                    
                            post_comments.append(comment_data)
                            unique_comment_ids.add(comment['comment_id'])
                    condensed_data.append(post_comments)
    
    for lst in condensed_data:
        cond_data_cnt += len(lst)    
    
    print(f'post_cnt: {dup_post_cnt} + {unq_post_cnt} = {tot_post_cnt}')
    print(f'comm_cnt: {dup_comm_cnt} + {unq_comm_cnt} = {tot_comm_cnt}')
    print(f'cond_cnt: {cond_data_cnt} = {unq_post_cnt} + {unq_comm_cnt}')
    print(f'Cross_ck: {len(unq_reddit_post_ids)} = {unq_post_cnt}')
    
    return condensed_data, unique_post_ids, unique_comment_ids

