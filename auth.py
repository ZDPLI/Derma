import json
import os
from hashlib import sha256

USERS_FILE = os.path.join('data', 'users.json')


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2)


class AuthManager:
    def __init__(self):
        self.users = load_users()

    @staticmethod
    def _hash(pw: str) -> str:
        return sha256(pw.encode('utf-8')).hexdigest()

    def authenticate(self, username: str, password: str):
        hashed = self._hash(password)
        for u in self.users:
            if u['username'] == username and u['password'] == hashed:
                return u
        return None

    def add_user(self, username: str, password: str, is_admin: bool = False, subscription: bool = True) -> bool:
        if any(u['username'] == username for u in self.users):
            return False
        user = {
            'username': username,
            'password': self._hash(password),
            'is_admin': is_admin,
            'subscription': subscription,
        }
        self.users.append(user)
        save_users(self.users)
        return True

    def toggle_subscription(self, username: str):
        for u in self.users:
            if u['username'] == username:
                u['subscription'] = not u['subscription']
                save_users(self.users)
                return True
        return False

    def get_users_info(self):
        return [
            {'username': u['username'], 'subscription': u['subscription'], 'is_admin': u['is_admin']}
            for u in self.users
        ]

