from typing import List, Dict, Any
from datetime import datetime
from jet.data.utils import generate_unique_hash


class RedditPostExtractor:
    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    def extract_most_upvoted_conversation_chain(self) -> List[Dict[str, Any]]:
        poster_data = self._extract_poster_data([self.data[0]])

        if len(self.data) < 2 or not self.data[1]:
            return [poster_data]

        # Extract comments and sort them by score
        poster_author = poster_data['author']
        comments_data = self._extract_comments(
            self.data[1:], poster_author, is_reply=False)

        # Compile the conversation chain starting from the post data
        conversation_chain = []

        # most_upvoted_comment = self._get_most_upvoted_comment(comments_data)

        # If there are comments, append them and their replies recursively
        for comment in comments_data:
            self._append_conversation_chain(
                conversation_chain, comment)

        return {
            'poster': poster_data,
            'comments': comments_data,
        }

    def _get_most_upvoted_comment(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not comments:
            return {}
        return max(comments, key=lambda x: x['score'])

    def _extract_poster_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        post_details = data[0]['data']
        # Convert post_details['created_utc'] to datetime (sample format: 1708358887.0)
        created_date = datetime.utcfromtimestamp(
            post_details['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
        score = post_details.get('score', 0)

        return {
            'id': post_details.get('id', generate_unique_hash()),
            'author': post_details.get('author', None),
            'type': 'post',
            'is_poster': True,
            'is_top_level': True,
            'score': score,
            'is_negative': score < 0,
            'title': post_details.get('title', ''),
            'text': post_details.get('selftext', ''),
            'created': created_date
        }

    def _extract_comments(self, comments: List[Dict[str, Any]], poster_author: str, is_reply: bool = False) -> List[Dict[str, Any]]:
        # Find comment with the most score

        results = []
        for comment in comments:
            comment_data = comment['data']
            if 'data' in comment and 'body' in comment_data:
                comment_author = comment_data.get('author', None)
                is_poster = comment_author == poster_author
                created_date = datetime.utcfromtimestamp(
                    comment_data['created_utc']).strftime('%Y-%m-%d %H:%M:%S')
                score = comment_data.get('score', 0)
                obj = {
                    'id': comment_data.get('id', generate_unique_hash()),
                    'permalink': comment_data.get("permalink", ""),
                    'author': comment_author,
                    'type': 'comment' if not is_reply else 'reply',
                    'is_poster': is_poster,
                    'is_top_level': False,
                    'text': comment_data.get('body', ''),
                    'score': score,
                    'is_negative': score < 0,
                    'replies': self._extract_replies(comment_data.get('replies', {}), poster_author),
                    'is_reply': is_reply,
                    'created': created_date
                }
                results.append(obj)

        sorted_comments = sorted(
            results, key=lambda x: x['score'], reverse=True)
        return sorted_comments

    def _extract_replies(self, replies: Dict[str, Any], poster_author: str) -> List[Dict[str, Any]]:
        if replies and 'data' in replies and 'children' in replies['data']:
            return self._extract_comments(replies['data']['children'], poster_author, is_reply=True)
        return []

    def _append_conversation_chain(self, conversation_chain: List[Dict[str, Any]], comment: Dict[str, Any]) -> None:
        # Append the comment or reply itself
        comment_replies = comment['replies']
        conversation = comment.copy()
        del conversation['replies']

        conversation_chain.append(conversation)

        # most_upvoted_reply = self._get_most_upvoted_comment(comment_replies)

        for reply in comment_replies:
            self._append_conversation_chain(
                conversation_chain, reply)
