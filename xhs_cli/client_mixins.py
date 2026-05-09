"""Domain-specific endpoint mixins for XhsClient."""

from __future__ import annotations

import json
import logging
import mimetypes
import random
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .constants import CREATOR_HOST, HOME_URL, UPLOAD_HOST, USER_AGENT
from .cookies import (
    cache_note_context,
    cookies_to_string,
    get_cached_note_context,
    get_config_dir,
    invalidate_note_context,
)
from .exceptions import NeedVerifyError, UnsupportedOperationError, XhsApiError
from .html_parser import extract_note_from_html

logger = logging.getLogger(__name__)

_SEARCH_DEFAULT_FILTERS = [
    {"tags": ["general"], "type": "sort_type"},
    {"tags": ["不限"], "type": "filter_note_type"},
    {"tags": ["不限"], "type": "filter_note_time"},
    {"tags": ["不限"], "type": "filter_note_range"},
    {"tags": ["不限"], "type": "filter_pos_distance"},
]
_SEARCH_SESSION_TTL_SECONDS = 600
_SEARCH_SESSION_MAX_SIZE = 128
_SEARCH_SESSION_LOCK = threading.RLock()
_SEARCH_SESSION_CACHE: OrderedDict[tuple[str, str, int], dict[str, Any]] = OrderedDict()
_SEARCH_SESSION_CACHE_PATH: Path | None = None
_SEARCH_SESSION_CACHE_LOADED = False


def _generate_search_id() -> str:
    """Generate a unique search ID (base36 of timestamp << 64 + random)."""
    e = int(time.time() * 1000) << 64
    t = random.randint(0, 2147483646)
    num = e + t

    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if num == 0:
        return "0"
    result = ""
    while num > 0:
        result = alphabet[num % 36] + result
        num //= 36
    return result


def _search_session_key(keyword: str, sort: str, note_type: int) -> tuple[str, str, int]:
    return (keyword.strip(), sort, note_type)


def _search_session_path() -> Path:
    return get_config_dir() / "search_sessions.json"


def _serialize_search_session_key(key: tuple[str, str, int]) -> str:
    return json.dumps([key[0], key[1], key[2]], ensure_ascii=False)


def _deserialize_search_session_key(value: str) -> tuple[str, str, int] | None:
    try:
        keyword, sort, note_type = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(keyword, str) or not isinstance(sort, str):
        return None
    try:
        normalized_note_type = int(note_type)
    except (TypeError, ValueError):
        return None
    return (keyword, sort, normalized_note_type)


def _load_search_session_cache_from_disk(path: Path) -> OrderedDict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        return OrderedDict()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return OrderedDict()
    if not isinstance(data, dict):
        return OrderedDict()

    normalized: list[tuple[tuple[str, str, int], dict[str, Any]]] = []
    for raw_key, value in data.items():
        key = _deserialize_search_session_key(raw_key)
        if not key or not isinstance(value, dict):
            continue
        if not value.get("search_id"):
            continue
        normalized.append((key, {
            "search_id": str(value["search_id"]),
            "created_at": float(value.get("created_at", 0) or 0),
            "last_used_at": float(value.get("last_used_at", 0) or 0),
        }))
    normalized.sort(key=lambda item: float(item[1].get("last_used_at", 0)))
    return OrderedDict(normalized)


def _save_search_session_cache(path: Path) -> None:
    payload = OrderedDict(
        (
            _serialize_search_session_key(key),
            dict(value),
        )
        for key, value in _SEARCH_SESSION_CACHE.items()
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    path.chmod(0o600)


def _ensure_search_session_cache_loaded() -> None:
    global _SEARCH_SESSION_CACHE_LOADED, _SEARCH_SESSION_CACHE_PATH, _SEARCH_SESSION_CACHE
    path = _search_session_path()
    if _SEARCH_SESSION_CACHE_LOADED and _SEARCH_SESSION_CACHE_PATH == path:
        return
    _SEARCH_SESSION_CACHE = _load_search_session_cache_from_disk(path)
    _SEARCH_SESSION_CACHE_PATH = path
    _SEARCH_SESSION_CACHE_LOADED = True


def _prune_search_sessions(now: float) -> None:
    expired_keys = [
        key
        for key, value in _SEARCH_SESSION_CACHE.items()
        if now - float(value.get("last_used_at", 0)) > _SEARCH_SESSION_TTL_SECONDS
    ]
    for key in expired_keys:
        _SEARCH_SESSION_CACHE.pop(key, None)

    while len(_SEARCH_SESSION_CACHE) > _SEARCH_SESSION_MAX_SIZE:
        _SEARCH_SESSION_CACHE.popitem(last=False)


def _acquire_search_session(keyword: str, sort: str, note_type: int) -> tuple[str, bool]:
    now = time.time()
    key = _search_session_key(keyword, sort, note_type)

    with _SEARCH_SESSION_LOCK:
        _ensure_search_session_cache_loaded()
        _prune_search_sessions(now)
        existing = _SEARCH_SESSION_CACHE.get(key)
        if existing:
            existing["last_used_at"] = now
            _SEARCH_SESSION_CACHE.move_to_end(key)
            _save_search_session_cache(_SEARCH_SESSION_CACHE_PATH or _search_session_path())
            return str(existing["search_id"]), False

        search_id = _generate_search_id()
        _SEARCH_SESSION_CACHE[key] = {
            "search_id": search_id,
            "created_at": now,
            "last_used_at": now,
        }
        _save_search_session_cache(_SEARCH_SESSION_CACHE_PATH or _search_session_path())
        return search_id, True


def get_search_session_stats() -> dict[str, Any]:
    """Return lightweight debug stats for the in-memory search session cache."""
    now = time.time()
    with _SEARCH_SESSION_LOCK:
        _ensure_search_session_cache_loaded()
        _prune_search_sessions(now)
        if not _SEARCH_SESSION_CACHE:
            return {
                "active_count": 0,
                "last_keyword": "",
                "last_sort": "",
                "last_note_type": None,
            }

        last_key = next(reversed(_SEARCH_SESSION_CACHE))
        return {
            "active_count": len(_SEARCH_SESSION_CACHE),
            "last_keyword": last_key[0],
            "last_sort": last_key[1],
            "last_note_type": last_key[2],
        }


class ReadingEndpointsMixin:
    """Read-only note, profile, and discovery endpoints."""

    def _search_request_id(self) -> str:
        return f"{random.randint(1_000_000_000, 2_147_483_647)}-{int(time.time() * 1000)}"

    def _fetch_note_html(
        self,
        note_id: str,
        xsec_token: str = "",
        xsec_source: str = "pc_feed",
    ) -> str:
        if xsec_token:
            url = f"{HOME_URL}/explore/{note_id}?xsec_token={xsec_token}&xsec_source={xsec_source}"
        else:
            url = f"{HOME_URL}/explore/{note_id}"

        resp = self._request_with_retry(
            "GET",
            url,
            headers={
                "user-agent": USER_AGENT,
                "referer": f"{HOME_URL}/",
                "cookie": cookies_to_string(self.cookies),
            },
        )
        return resp.text

    def resolve_xsec_context(
        self,
        note_id: str,
        preferred_token: str = "",
        preferred_source: str = "",
    ) -> tuple[str, str]:
        """Resolve xsec_token/xsec_source from input, cache, or note page metadata."""
        if preferred_token:
            cache_note_context(note_id, preferred_token, preferred_source)
            return preferred_token, preferred_source

        cached = get_cached_note_context(note_id)
        if cached.get("token"):
            return cached["token"], cached.get("source", "")

        html = self._fetch_note_html(note_id)
        patterns = [
            r'"xsec_token"\s*:\s*"([^"]+)"',
            r"xsec_token=([^&\"']+)",
            r"'xsec_token':'([^']+)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                token = match.group(1)
                source_match = re.search(r"xsec_source=([^&\"']+)", html)
                source = source_match.group(1) if source_match else preferred_source
                cache_note_context(note_id, token, source)
                return token, source
        return "", preferred_source

    def resolve_xsec_token(self, note_id: str, preferred_token: str = "") -> str:
        """Resolve xsec_token from explicit input, cache, or note page metadata."""
        token, _source = self.resolve_xsec_context(note_id, preferred_token)
        return token

    def get_self_info(self) -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/v2/user/me")

    def get_user_info(self, user_id: str) -> dict[str, Any]:
        try:
            return self._main_api_get("/api/sns/web/v1/user/otherinfo", {
                "target_user_id": user_id,
            })
        except XhsApiError:
            # Fallback: extract user info from profile HTML SSR data
            return self._get_user_info_from_html(user_id)

    def get_user_notes(self, user_id: str, cursor: str = "") -> dict[str, Any]:
        user_id = self._resolve_user_id(user_id)
        # Fetch profile SSR to get xsec_token and notes as fallback
        ssr = None
        try:
            ssr = self._fetch_user_profile_ssr(user_id)
            notes_groups = ssr.get("user", {}).get("notes", [])
            raw_notes = notes_groups[0] if notes_groups else []
            xsec_token = ""
            if raw_notes:
                xsec_token = raw_notes[0].get("xsecToken", "") or raw_notes[0].get("noteCard", {}).get("xsecToken", "")
            if xsec_token:
                result = self._main_api_get("/api/sns/web/v1/user_posted", {
                    "num": 30,
                    "cursor": cursor,
                    "user_id": user_id,
                    "image_formats": "jpg,webp,avif",
                    "xsec_token": xsec_token,
                    "xsec_source": "pc_user",
                })
                return result
        except XhsApiError:
            pass
        # Fallback: parse notes from SSR data, but only if they have valid note_ids
        if ssr:
            result = self._parse_notes_from_ssr(ssr)
            notes = result.get("notes", [])
            if notes and notes[0].get("note_id"):
                return result
        # If SSR data is incomplete (e.g. no login), raise error
        raise XhsApiError(-1, "Failed to get user notes (cookie may be expired)")

    def _resolve_user_id(self, user_id: str) -> str:
        """Resolve a red_id or nickname to internal hex user_id if needed."""
        # Already a hex user_id
        if re.fullmatch(r"[0-9a-f]{24}", user_id):
            return user_id
        # Try to resolve via search_users
        result = self.search_users(user_id)
        for dto in result.get("user_info_dtos", []):
            base = dto.get("user_base_dto", {})
            if base.get("red_id") == user_id or base.get("user_nickname") == user_id:
                return base["user_id"]
        # If only one result, use it
        dtos = result.get("user_info_dtos", [])
        if len(dtos) == 1:
            return dtos[0].get("user_base_dto", {}).get("user_id", user_id)
        return user_id

    def _fetch_user_profile_ssr(self, user_id: str) -> dict[str, Any]:
        """Fetch user profile page and extract __INITIAL_STATE__ SSR data."""
        user_id = self._resolve_user_id(user_id)
        url = f"{HOME_URL}/user/profile/{user_id}"
        resp = self._request_with_retry("GET", url, headers={
            "user-agent": USER_AGENT,
            "referer": f"{HOME_URL}/",
            "cookie": cookies_to_string(self.cookies),
        })
        match = re.search(r"<script>window\.__INITIAL_STATE__=(.*?)</script>", resp.text, re.DOTALL)
        if not match:
            raise XhsApiError(-1, "Failed to extract SSR data from user profile page")
        raw = match.group(1).replace("undefined", "null")
        return json.loads(raw)

    def _get_user_info_from_html(self, user_id: str) -> dict[str, Any]:
        """Extract user info from profile HTML SSR data."""
        ssr = self._fetch_user_profile_ssr(user_id)
        page_data = ssr.get("user", {}).get("userPageData", {})
        basic_info = page_data.get("basicInfo", {})
        interactions = page_data.get("interactions", [])
        tags = page_data.get("tags", [])
        return {
            "basic_info": {
                "nickname": basic_info.get("nickname", ""),
                "red_id": basic_info.get("redId", ""),
                "desc": basic_info.get("desc", ""),
                "gender": basic_info.get("gender", 0),
                "ip_location": basic_info.get("ipLocation", ""),
                "image": basic_info.get("imageb", basic_info.get("image", "")),
            },
            "interactions": interactions,
            "tags": tags,
        }

    def _parse_notes_from_ssr(self, ssr: dict[str, Any]) -> dict[str, Any]:
        """Parse notes from already-fetched SSR data."""
        notes_groups = ssr.get("user", {}).get("notes", [])
        raw_notes = notes_groups[0] if notes_groups else []
        notes = []
        for item in raw_notes:
            nc = item.get("noteCard", {})
            interact = nc.get("interactInfo", {})
            cover = nc.get("cover", {})
            cover_url = ""
            for img in cover.get("infoList", []):
                if img.get("imageScene") == "WB_PRV":
                    cover_url = img.get("url", "")
                    break
            notes.append({
                "note_id": item.get("id", "") or nc.get("noteId", ""),
                "display_title": nc.get("displayTitle", ""),
                "type": nc.get("type", ""),
                "user": nc.get("user", {}),
                "liked_count": interact.get("likedCount", "0"),
                "liked": interact.get("liked", False),
                "sticky": interact.get("sticky", False),
                "cover_url": cover_url,
                "xsec_token": nc.get("xsecToken", item.get("xsecToken", "")),
            })
        return {
            "notes": notes,
            "has_more": False,
            "cursor": "",
        }

    def _get_user_notes_from_html(self, user_id: str) -> dict[str, Any]:
        """Extract user notes from profile HTML SSR data."""
        ssr = self._fetch_user_profile_ssr(user_id)
        return self._parse_notes_from_ssr(ssr)

    def get_user_notes_with_detail(self, user_id: str) -> dict[str, Any]:
        """Get user notes with full content by fetching each note's detail."""
        listing = self.get_user_notes(user_id)
        notes = listing.get("notes", [])
        detailed_notes = []
        for note in notes:
            note_id = note.get("note_id", "")
            xsec_token = note.get("xsec_token", note.get("xsecToken", ""))
            if not note_id:
                continue
            try:
                detail = self.get_note_detail(note_id, xsec_token=xsec_token, xsec_source="pc_user")
                # Extract note_card from feed API response
                note_card = detail
                if isinstance(detail, dict):
                    items = detail.get("items", [])
                    if items:
                        note_card = items[0].get("note_card", detail)
                    elif "note_card" in detail:
                        note_card = detail["note_card"]
                note_card["note_id"] = note_id
                detailed_notes.append(note_card)
            except Exception as exc:
                logger.debug("Failed to fetch detail for %s: %s", note_id, exc)
                detailed_notes.append(note)
        return {
            "notes": detailed_notes,
            "has_more": listing.get("has_more", False),
            "cursor": listing.get("cursor", ""),
        }

    def search_notes(
        self,
        keyword: str,
        page: int = 1,
        page_size: int = 20,
        sort: str = "general",
        note_type: int = 0,
    ) -> Any:
        search_id, is_new_session = _acquire_search_session(keyword, sort, note_type)
        if is_new_session:
            request_id = self._search_request_id()
            try:
                self._main_api_post("/api/sns/web/v1/search/onebox", {
                    "keyword": keyword,
                    "search_id": search_id,
                    "biz_type": "web_search_user",
                    "request_id": request_id,
                })
                self._main_api_get("/api/sns/web/v1/search/filter", {
                    "keyword": keyword,
                    "search_id": search_id,
                })
            except XhsApiError as exc:
                logger.debug("Search prewarm failed, continuing with search/notes: %s", exc)

        result = self._main_api_post("/api/sns/web/v1/search/notes", {
            "keyword": keyword,
            "page": page,
            "page_size": page_size,
            "search_id": search_id,
            "sort": sort,
            "note_type": note_type,
            "ext_flags": [],
            "filters": _SEARCH_DEFAULT_FILTERS,
            "geo": "",
            "image_formats": ["jpg", "webp", "avif"],
        })
        if is_new_session:
            try:
                self._main_api_get("/api/sns/web/v1/search/recommend", {"keyword": keyword})
            except XhsApiError as exc:
                logger.debug("Search recommend prefetch failed: %s", exc)
        return result

    def get_note_by_id(
        self,
        note_id: str,
        xsec_token: str = "",
        xsec_source: str = "pc_feed",
    ) -> Any:
        if xsec_token:
            cache_note_context(note_id, xsec_token, xsec_source)
        return self._main_api_post("/api/sns/web/v1/feed", {
            "source_note_id": note_id,
            "image_formats": ["jpg", "webp", "avif"],
            "extra": {"need_body_topic": "1"},
            "xsec_source": xsec_source,
            "xsec_token": xsec_token,
        })

    def get_note_from_html(
        self,
        note_id: str,
        xsec_token: str = "",
        xsec_source: str = "pc_feed",
    ) -> dict[str, Any]:
        """Fetch note by parsing server-rendered HTML (no xsec_token required)."""
        html = self._fetch_note_html(note_id, xsec_token=xsec_token, xsec_source=xsec_source)
        return extract_note_from_html(html, note_id)

    def get_note_detail(
        self,
        note_id: str,
        xsec_token: str = "",
        xsec_source: str = "",
    ) -> dict[str, Any]:
        """Read a note via the best available channel.

        Strategy:
          - Has xsec_token → try feed API first, fall back to HTML on error
          - No xsec_token  → go straight to HTML (feed API would reject)
        """
        cached = get_cached_note_context(note_id)
        token = xsec_token or cached.get("token", "")
        source = xsec_source or cached.get("source", "") or "pc_feed"
        used_cached_context = not xsec_token and bool(cached.get("token"))
        if token:
            try:
                return self.get_note_by_id(note_id, xsec_token=token, xsec_source=source)
            except (NeedVerifyError, XhsApiError) as exc:
                logger.info("Feed API failed (%s), falling back to HTML", exc)
                if used_cached_context:
                    invalidate_note_context(note_id)
                    token = ""
        return self.get_note_from_html(note_id, xsec_token=token or "", xsec_source=source)

    def get_home_feed(self, category: str = "homefeed_recommend") -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/homefeed", {
            "cursor_score": "",
            "num": 40,
            "refresh_type": 1,
            "note_index": 0,
            "unread_begin_note_id": "",
            "unread_end_note_id": "",
            "unread_note_count": 0,
            "category": category,
            "search_key": "",
            "need_num": 40,
            "image_scenes": ["FD_PRV_WEBP", "FD_WM_WEBP"],
        })

    def get_hot_feed(self, category: str = "homefeed.fashion_v3") -> dict[str, Any]:
        return self.get_home_feed(category=category)

    def get_comments(
        self,
        note_id: str,
        cursor: str = "",
        xsec_token: str = "",
        top_comment_id: str = "",
        xsec_source: str = "",
    ) -> Any:
        cached = get_cached_note_context(note_id)
        used_cached_context = not xsec_token and bool(cached.get("token"))
        token, source = self.resolve_xsec_context(note_id, xsec_token, xsec_source)
        if not token:
            raise XhsApiError(
                "Could not resolve xsec_token for comments. Pass a full note URL or --xsec-token explicitly."
            )
        if source:
            cache_note_context(note_id, token, source)
        try:
            return self._main_api_get("/api/sns/web/v2/comment/page", {
                "note_id": note_id,
                "cursor": cursor,
                "top_comment_id": top_comment_id,
                "image_formats": "jpg,webp,avif",
                "xsec_token": token,
            })
        except (NeedVerifyError, XhsApiError):
            if not used_cached_context:
                raise
            invalidate_note_context(note_id)
            refreshed_token, refreshed_source = self.resolve_xsec_context(note_id, "", xsec_source)
            if not refreshed_token:
                raise
            if refreshed_source:
                cache_note_context(note_id, refreshed_token, refreshed_source)
            return self._main_api_get("/api/sns/web/v2/comment/page", {
                "note_id": note_id,
                "cursor": cursor,
                "top_comment_id": top_comment_id,
                "image_formats": "jpg,webp,avif",
                "xsec_token": refreshed_token,
            })

    def get_all_comments(
        self,
        note_id: str,
        xsec_token: str = "",
        xsec_source: str = "",
        max_pages: int = 20,
    ) -> dict[str, Any]:
        all_comments: list[dict[str, Any]] = []
        cursor = ""
        pages = 0

        while pages < max_pages:
            data = self.get_comments(
                note_id,
                cursor=cursor,
                xsec_token=xsec_token,
                xsec_source=xsec_source,
            )
            if not isinstance(data, dict):
                break

            comments = data.get("comments", [])
            all_comments.extend(comments)
            pages += 1

            has_more = data.get("has_more", False)
            next_cursor = data.get("cursor", "")
            if not has_more or not next_cursor:
                break
            cursor = next_cursor

        return {
            "comments": all_comments,
            "has_more": False,
            "cursor": "",
            "total_fetched": len(all_comments),
            "pages_fetched": pages,
        }

    def get_sub_comments(
        self,
        note_id: str,
        root_comment_id: str,
        num: int = 30,
        cursor: str = "",
    ) -> Any:
        return self._main_api_get("/api/sns/web/v2/comment/sub/page", {
            "note_id": note_id,
            "root_comment_id": root_comment_id,
            "num": num,
            "cursor": cursor,
        })


class InteractionEndpointsMixin:
    """Mutating note interaction endpoints."""

    def post_comment(self, note_id: str, content: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/comment/post", {
            "note_id": note_id,
            "content": content,
            "at_users": [],
        })

    def reply_comment(self, note_id: str, target_comment_id: str, content: str) -> Any:
        return self._main_api_post("/api/sns/web/v1/comment/post", {
            "note_id": note_id,
            "content": content,
            "target_comment_id": target_comment_id,
            "at_users": [],
        })

    def like_note(self, note_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/note/like", {"note_oid": note_id})

    def unlike_note(self, note_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/note/dislike", {"note_oid": note_id})

    def favorite_note(self, note_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/note/collect", {"note_id": note_id})

    def unfavorite_note(self, note_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/note/uncollect", {"note_ids": note_id})

    def delete_comment(self, note_id: str, comment_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/comment/delete", {
            "note_id": note_id,
            "comment_id": comment_id,
        })


class CreatorEndpointsMixin:
    """Creator platform search, upload, and publishing endpoints."""

    def search_topics(self, keyword: str) -> dict[str, Any]:
        return self._creator_post("/web_api/sns/v1/search/topic", {
            "keyword": keyword,
            "suggest_topic_request": {"title": "", "desc": ""},
            "page": {"page_size": 20, "page": 1},
        })

    def search_users(self, keyword: str) -> dict[str, Any]:
        return self._creator_post("/web_api/sns/v1/search/user_info", {
            "keyword": keyword,
            "search_id": str(int(time.time() * 1000)),
            "page": {"page_size": 20, "page": 1},
        })

    def get_upload_permit(self, file_type: str = "image", count: int = 1) -> dict[str, str]:
        data = self._creator_get("/api/media/v1/upload/web/permit", {
            "biz_name": "spectrum",
            "scene": file_type,
            "file_count": count,
            "version": 1,
            "source": "web",
        })
        permit = data["uploadTempPermits"][0]
        return {"fileId": permit["fileIds"][0], "token": permit["token"]}

    def upload_file(
        self,
        file_id: str,
        token: str,
        file_path: str,
        content_type: str | None = None,
    ) -> None:
        with open(file_path, "rb") as f:
            file_data = f.read()

        url = f"{UPLOAD_HOST}/{file_id}"
        content_type = content_type or mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        resp = self._request_with_retry(
            "PUT",
            url,
            headers={
                "X-Cos-Security-Token": token,
                "Content-Type": content_type,
            },
            content=file_data,
        )
        if resp.status_code >= 400:
            raise XhsApiError(f"Upload failed: {resp.status_code} {resp.reason_phrase}")

    def create_image_note(
        self,
        title: str,
        desc: str,
        image_file_ids: list[str],
        topics: list[dict[str, str]] | None = None,
        is_private: bool = False,
    ) -> Any:
        images = [{"file_id": fid, "metadata": {"source": -1}} for fid in image_file_ids]
        business_binds = {
            "version": 1,
            "noteId": 0,
            "noteOrderBind": {},
            "notePostTiming": {"postTime": None},
            "noteCollectionBind": {"id": ""},
        }
        data = {
            "common": {
                "type": "normal",
                "title": title,
                "note_id": "",
                "desc": desc,
                "source": '{"type":"web","ids":"","extraInfo":"{\\"subType\\":\\"official\\"}"}',
                "business_binds": json.dumps(business_binds),
                "ats": [],
                "hash_tag": topics or [],
                "post_loc": {},
                "privacy_info": {"op_type": 1, "type": 1 if is_private else 0},
            },
            "image_info": {"images": images},
            "video_info": None,
        }
        return self._main_api_post("/web_api/sns/v2/note", data, {
            "origin": CREATOR_HOST,
            "referer": f"{CREATOR_HOST}/",
        })

    def delete_note(self, note_id: str) -> dict[str, Any]:
        try:
            return self._creator_post("/api/galaxy/creator/note/delete", {
                "note_id": note_id,
            })
        except XhsApiError as exc:
            response = exc.response if isinstance(exc.response, dict) else {}
            if response.get("status") == 404 or "404" in str(exc):
                raise UnsupportedOperationError(
                    "Delete note is currently unavailable from the public web API. "
                    "The command remains experimental until the new endpoint is re-captured."
                ) from None
            raise

    def get_creator_note_list(self, tab: int = 0, page: int = 0) -> dict[str, Any]:
        return self._creator_get("/api/galaxy/v2/creator/note/user/posted", {
            "tab": tab,
            "page": page,
        })


class SocialEndpointsMixin:
    """Social graph and saved-content endpoints."""

    def follow_user(self, user_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/user/follow", {"target_user_id": user_id})

    def unfollow_user(self, user_id: str) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/user/unfollow", {"target_user_id": user_id})

    def get_user_favorites(self, user_id: str, cursor: str = "") -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/v2/note/collect/page", {
            "user_id": user_id,
            "cursor": cursor,
            "num": 30,
        })


class NotificationEndpointsMixin:
    """Notification and unread-count endpoints."""

    def get_unread_count(self) -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/unread_count", {})

    def get_notification_mentions(self, cursor: str = "", num: int = 20) -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/v1/you/mentions", {
            "num": num,
            "cursor": cursor,
        })

    def get_notification_likes(self, cursor: str = "", num: int = 20) -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/v1/you/likes", {
            "num": num,
            "cursor": cursor,
        })

    def get_notification_connections(self, cursor: str = "", num: int = 20) -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/v1/you/connections", {
            "num": num,
            "cursor": cursor,
        })


class AuthEndpointsMixin:
    """Authentication-specific endpoints."""

    def login_activate(self) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/login/activate", {})

    def create_qr_login(self) -> dict[str, Any]:
        return self._main_api_post("/api/sns/web/v1/login/qrcode/create", {"qr_type": 1})

    def check_qr_status(self, qr_id: str, code: str) -> dict[str, Any]:
        return self._main_api_post("/api/qrcode/userinfo", {
            "qrId": qr_id,
            "code": code,
        }, {
            "service-tag": "webcn",
        })

    def complete_qr_login(self, qr_id: str, code: str) -> dict[str, Any]:
        return self._main_api_get("/api/sns/web/v1/login/qrcode/status", {
            "qr_id": qr_id,
            "code": code,
        })
