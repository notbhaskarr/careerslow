import os
import json
import redis
from typing import Optional
from src.schemas.document_schemas import ParsedJD
from src.utils.cache_keys import plan_cache_key


class RedisCache:
    """
    Redis cache for parsed documents, pair analyses, interview plans, and voice sessions.

    Key layout:
        jd:{jd_hash}              — parsed JD (shared across candidates)
        resume:{resume_id}        — parsed resume
        analysis:{pair_id}        — gap analysis for resume+JD pair
        interview_plan:{pair_id}:v{N} — mock interview plan
        transcript:{session_id}   — one voice session transcript
        session_meta:{session_id} — segment tracking for that session
        debrief:{session_id}      — cached post-interview debrief
        session_pair:{session_id} — pair_id lookup for a voice session
        extract:{resume_id}       — PDF extraction debug (raw text + page stats)
        chunks:{resume_id}        — Qdrant chunk payloads debug (section_type + text)
    """

    def __init__(self):
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self.client = redis.from_url(
                redis_url, decode_responses=True, socket_timeout=2.0
            )
        else:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            self.client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=2.0,
            )
        try:
            self.client.ping()
            print("Connected to Redis Cache!")
        except Exception as e:
            print(f"Warning: Failed to connect to Redis. {e}")

    def get_jd(self, jd_hash: str) -> Optional[ParsedJD]:
        """Load a cached ParsedJD by content hash."""
        try:
            cached_data = self.client.get(f"jd:{jd_hash}")
            if cached_data:
                print(f"Cache HIT for JD: {jd_hash}")
                return ParsedJD.model_validate_json(cached_data)
        except Exception as e:
            print(f"Redis get_jd error: {e}")
        return None

    def set_jd(self, jd_hash: str, parsed_jd: ParsedJD):
        """Store parsed JD under jd:{hash}."""
        try:
            self.client.set(f"jd:{jd_hash}", parsed_jd.model_dump_json())
        except Exception as e:
            print(f"Redis set_jd error: {e}")

    def get_resume(self, resume_id: str):
        """Load cached ParsedResume by resume_id."""
        from src.schemas.document_schemas import ParsedResume

        try:
            cached_data = self.client.get(f"resume:{resume_id}")
            if cached_data:
                print(f"Cache HIT for Resume: {resume_id}")
                return ParsedResume.model_validate_json(cached_data)
        except Exception as e:
            print(f"Redis get_resume error: {e}")
        return None

    def set_resume(self, resume_id: str, parsed_resume):
        """Store parsed resume under resume:{resume_id}."""
        try:
            self.client.set(f"resume:{resume_id}", parsed_resume.model_dump_json())
        except Exception as e:
            print(f"Redis set_resume error: {e}")

    def get_extraction(self, resume_id: str) -> Optional[dict]:
        """Load PDF extraction debug payload for a resume."""
        try:
            cached = self.client.get(f"extract:{resume_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get_extraction error: {e}")
        return None

    def set_extraction(self, resume_id: str, data: dict):
        """Store PDF extraction debug under extract:{resume_id}."""
        try:
            self.client.set(f"extract:{resume_id}", json.dumps(data))
        except Exception as e:
            print(f"Redis set_extraction error: {e}")

    def get_chunks(self, resume_id: str) -> Optional[dict]:
        """Load chunk debug payload for a resume."""
        try:
            cached = self.client.get(f"chunks:{resume_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get_chunks error: {e}")
        return None

    def set_chunks(self, resume_id: str, data: dict):
        """Store chunk debug under chunks:{resume_id}."""
        try:
            self.client.set(f"chunks:{resume_id}", json.dumps(data))
        except Exception as e:
            print(f"Redis set_chunks error: {e}")

    def get_analysis(self, pair_id: str):
        """Load gap analysis bundle for a resume+JD pair."""
        try:
            cached_data = self.client.get(f"analysis:{pair_id}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Redis get_analysis error: {e}")
        return None

    def set_analysis(self, pair_id: str, analysis_data: dict):
        """Store gap analysis under analysis:{pair_id}."""
        try:
            self.client.set(f"analysis:{pair_id}", json.dumps(analysis_data))
        except Exception as e:
            print(f"Redis set_analysis error: {e}")

    def get_interview_plan(self, pair_id: str):
        """Load interview plan for a pair (versioned by PLAN_VERSION)."""
        try:
            key = f"interview_plan:{plan_cache_key(pair_id)}"
            cached_data = self.client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"Redis get_interview_plan error: {e}")
        return None

    def set_interview_plan(self, pair_id: str, plan_data: dict):
        """Store versioned interview plan."""
        try:
            key = f"interview_plan:{plan_cache_key(pair_id)}"
            self.client.set(key, json.dumps(plan_data))
        except Exception as e:
            print(f"Redis set_interview_plan error: {e}")

    def get_transcript(self, session_id: str) -> str:
        """Load transcript for a voice session."""
        try:
            return self.client.get(f"transcript:{session_id}") or ""
        except Exception as e:
            print(f"Redis get_transcript error: {e}")
        return ""

    def set_transcript(self, session_id: str, transcript: str):
        """Save transcript for a voice session."""
        try:
            self.client.set(f"transcript:{session_id}", transcript)
        except Exception as e:
            print(f"Redis set_transcript error: {e}")

    def get_session_meta(self, session_id: str):
        """Load segment tracking metadata for a voice session."""
        try:
            cached = self.client.get(f"session_meta:{session_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get_session_meta error: {e}")
        return None

    def set_session_meta(self, session_id: str, meta: dict):
        """Save segment tracking metadata."""
        try:
            self.client.set(f"session_meta:{session_id}", json.dumps(meta))
        except Exception as e:
            print(f"Redis set_session_meta error: {e}")

    def get_debrief(self, session_id: str):
        """Load cached debrief for a voice session."""
        try:
            cached = self.client.get(f"debrief:{session_id}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"Redis get_debrief error: {e}")
        return None

    def set_debrief(self, session_id: str, debrief: dict):
        """Cache generated debrief."""
        try:
            self.client.set(f"debrief:{session_id}", json.dumps(debrief))
        except Exception as e:
            print(f"Redis set_debrief error: {e}")

    def link_session_pair(self, session_id: str, pair_id: str):
        """Record which analysis pair a voice session belongs to."""
        try:
            self.client.set(f"session_pair:{session_id}", pair_id)
        except Exception as e:
            print(f"Redis link_session_pair error: {e}")

    def get_session_pair(self, session_id: str) -> Optional[str]:
        """Resolve pair_id from session_id."""
        try:
            return self.client.get(f"session_pair:{session_id}")
        except Exception as e:
            print(f"Redis get_session_pair error: {e}")
        return None

    def _key_entry(self, key: str, preview_chars: int = 400) -> dict:
        """Build one inspect row for a Redis key (exists, type, size, preview)."""
        try:
            exists = bool(self.client.exists(key))
            if not exists:
                return {"key": key, "exists": False}
            key_type = self.client.type(key)
            ttl = self.client.ttl(key)
            raw = self.client.get(key) if key_type == "string" else None
            size = len(raw) if raw else 0
            preview = None
            parsed = None
            if raw:
                try:
                    parsed = json.loads(raw)
                    preview = json.dumps(parsed, indent=2)[:preview_chars]
                except json.JSONDecodeError:
                    preview = raw[:preview_chars]
            return {
                "key": key,
                "exists": True,
                "type": key_type,
                "ttl_seconds": ttl if ttl >= 0 else None,
                "size_bytes": size,
                "preview": preview,
                "parsed_summary": self._summarize_value(parsed) if parsed else None,
            }
        except Exception as e:
            return {"key": key, "exists": False, "error": str(e)}

    def _summarize_value(self, value) -> dict:
        """Extract high-signal fields from cached JSON for UI display."""
        if not isinstance(value, dict):
            return {}
        summary = {}
        for field in (
            "overall_fit_score",
            "job_title",
            "cached",
            "current_segment",
            "segment_index",
            "phase",
            "overall_readiness",
        ):
            if field in value:
                summary[field] = value[field]
        if "gap_analyses" in value and isinstance(value["gap_analyses"], list):
            summary["gap_count"] = len(value["gap_analyses"])
        if "study_topics" in value and isinstance(value["study_topics"], list):
            summary["study_topics_count"] = len(value["study_topics"])
        if "segments" in value and isinstance(value["segments"], list):
            summary["segment_count"] = len(value["segments"])
        return summary

    def inspect_pair(self, pair_id: str) -> dict:
        """
        Return Redis keys and previews for a resume+JD pair.

        Includes analysis, interview plan, and linked resume/JD parse caches.
        """
        from src.utils.cache_keys import plan_cache_key

        keys = [
            f"analysis:{pair_id}",
            f"interview_plan:{plan_cache_key(pair_id)}",
        ]
        resume_suffix = None
        jd_suffix = None
        if pair_id.startswith("pair_") and pair_id.count("_") >= 2:
            parts = pair_id.split("_", 2)
            if len(parts) == 3:
                resume_suffix, jd_suffix = parts[1], parts[2]
                keys.extend([
                    f"resume:res_{resume_suffix}",
                    f"jd:{jd_suffix}",
                    f"extract:res_{resume_suffix}",
                    f"chunks:res_{resume_suffix}",
                ])

        entries = [self._key_entry(k) for k in keys]
        return {
            "pair_id": pair_id,
            "keys": entries,
            "keys_present": sum(1 for e in entries if e.get("exists")),
        }

    def inspect_session(self, session_id: str) -> dict:
        """Return Redis keys and previews for a voice mock session."""
        pair_id = self.get_session_pair(session_id)
        keys = [
            f"transcript:{session_id}",
            f"session_meta:{session_id}",
            f"debrief:{session_id}",
            f"session_pair:{session_id}",
        ]
        entries = [self._key_entry(k) for k in keys]
        result = {
            "session_id": session_id,
            "pair_id": pair_id,
            "keys": entries,
            "keys_present": sum(1 for e in entries if e.get("exists")),
        }
        if pair_id:
            result["pair_inspect"] = self.inspect_pair(pair_id)
        return result
