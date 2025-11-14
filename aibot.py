# ingres_assistant.py
import os
import time
import json
import random
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import pyodbc
import google.generativeai as genai

# -------------------------
# Config & environment
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
INGRES_ODBC_CONN = os.getenv("INGRES_ODBC_CONN", "")  # e.g. "DSN=ingres;UID=user;PWD=pass"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")  # used for admin actions in UI
READONLY_BY_DEFAULT = os.getenv("READONLY_BY_DEFAULT", "1") == "1"

if not GEMINI_API_KEY:
    st.error("Server misconfigured: GEMINI_API_KEY not found in environment.")
    st.stop()

if not INGRES_ODBC_CONN:
    st.warning("No INGRES_ODBC_CONN found - DB operations disabled. Set env var for database access if available.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# -------------------------
# Logging
# -------------------------
LOG_FILE = "assistant_logs.jsonl"
logger = logging.getLogger("ingres_assistant")
logger.setLevel(logging.INFO)

# Helper to append logs (structured)
def append_log(entry: Dict[str, Any]):
    entry["ts"] = datetime.utcnow().isoformat()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.exception("Could not write log: %s", e)

# -------------------------
# DB Interaction module
# -------------------------
def get_db_conn():
    if not INGRES_ODBC_CONN:
        raise RuntimeError("No INGRES_ODBC_CONN provided")
    return pyodbc.connect(INGRES_ODBC_CONN, autocommit=False)

def run_select_query(sql: str, params: Tuple = ()):
    """Execute a SELECT and return DataFrame."""
    conn = get_db_conn()
    try:
        df = pd.read_sql(sql, conn, params=list(params))
        return df
    finally:
        conn.close()

def run_write_query(sql: str, params: Tuple = ()):
    """Execute INSERT/UPDATE/DELETE, return rows affected."""
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rowcount = cur.rowcount
        conn.commit()
        return rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# -------------------------
# Simple SQL safety validator
# -------------------------
DISALLOWED_TOKENS = ["DROP", "TRUNCATE", "ALTER", "EXEC", "CREATE", "GRANT", "REVOKE"]
WRITE_TOKENS = ["INSERT", "UPDATE", "DELETE"]

def validate_sql(sql: str, allow_write: bool = False) -> Tuple[bool, str]:
    """
    Returns (ok, message). Prevents dangerous statements.
    """
    upper = sql.upper()
    for tok in DISALLOWED_TOKENS:
        if tok in upper:
            return False, f"Blocked token detected: {tok}"
    if any(w in upper for w in WRITE_TOKENS) and not allow_write:
        return False, "Write operations (INSERT/UPDATE/DELETE) are not allowed in read-only mode."
    return True, "OK"

# -------------------------
# NLP Intent module (Gemini)
# -------------------------
RETRY_COUNT = 3
RETRY_SLEEP = 1.0

def classify_intent_and_sql(nl: str, system_context: str = "") -> Dict[str, Any]:
    """
    Use Gemini to: classify intent (retrieve/insert/update/delete/admin), propose a parameterized SQL,
    and return extracted parameters as JSON.
    We keep a strict prompt to ask for a JSON response: {"intent": "...", "sql": "...", "params": {...}}
    """
    prompt = f"""
You are an intelligent assistant that translates plain English into safe, parameterized SQL for an Ingres database.
Return ONLY a JSON object with exactly three fields:
- "intent": one of ["retrieve","insert","update","delete","admin","other"]
- "sql": the parameterized SQL using ? placeholders for values (use LIMIT 100 for selects by default)
- "params": an ordered list of parameter values (strings/numbers) matching the placeholders, or [] if none.

Rules:
* Do NOT include any disallowed operations (DROP, TRUNCATE, ALTER, CREATE, GRANT, REVOKE).
* For SELECT queries, prefer explicit column names if user mentions them; else use SELECT * with LIMIT 100.
* For ambiguous destructive queries, return intent "other" and an empty sql and params.
* Keep JSON minimal and valid.

Context:
{system_context}

User request:
\"\"\"{nl}\"\"\"
"""
    # call gemini
    for attempt in range(RETRY_COUNT):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            session = model.start_chat(history=[])
            resp = session.send_message(prompt)
            text = resp.text.strip()
            # Try to parse JSON from response
            # Some model responses may include extra text — attempt to find the JSON substring
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = text[start:end]
            else:
                json_str = text
            parsed = json.loads(json_str)
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP * (attempt + 1))
    # fallback
    raise last_err

# -------------------------
# Response generation helper
# -------------------------
def format_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows returned."
    # Show top 10 rows
    return df.head(10).to_markdown(index=False)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Ingres Virtual Assistant", layout="wide")
st.title("Ingres Virtual Assistant — NL → SQL")

# sidebar: admin toggles (admin secret required)
with st.sidebar:
    st.header("Admin")
    admin_key = st.text_input("Admin secret (hidden)", type="password")
    is_admin = (admin_key != "" and admin_key == ADMIN_SECRET)
    if is_admin:
        st.success("Admin mode ON")
    else:
        if admin_key:
            st.error("Admin secret incorrect")
    read_only_mode = st.checkbox("Read-only mode", value=READONLY_BY_DEFAULT)
    st.markdown("---")
    st.markdown("Logs:")
    if st.button("Download logs"):
        try:
            with open(LOG_FILE, "rb") as f:
                st.download_button("Download logs file", f, file_name="assistant_logs.jsonl")
        except Exception:
            st.error("No logs available yet.")

# main chat area
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Ask the database (plain English)", height=150, placeholder="e.g. Show me students with GPA > 8.5")
    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Write something please.")
        else:
            # log user query (do not log raw sensitive data in production)
            append_log({"event": "user_query", "text": user_input[:1000]})
            with st.spinner("Interpreting your request..."):
                try:
                    parsed = classify_intent_and_sql(user_input)
                except Exception as e:
                    st.error(f"Could not parse the request: {e}")
                    append_log({"event": "nl_error", "error": str(e)})
                    parsed = {"intent":"other","sql":"","params":[]}

            intent = parsed.get("intent", "other")
            proposed_sql = parsed.get("sql", "")
            params = parsed.get("params", [])

            # Validate SQL
            allow_write = is_admin and not read_only_mode
            ok, msg = validate_sql(proposed_sql, allow_write=allow_write)
            if not ok:
                st.warning(f"Query blocked: {msg}")
                append_log({"event":"blocked_query","reason":msg,"intent":intent,"sql":proposed_sql})
            else:
                st.markdown("**Proposed query:**")
                st.code(proposed_sql)
                st.write("Parameters:", params)
                # Confirm before writes if not admin
                if any(tok in proposed_sql.upper() for tok in WRITE_TOKENS):
                    if not is_admin:
                        st.error("Write operations require admin privileges.")
                        append_log({"event":"write_blocked","sql":proposed_sql})
                    else:
                        if st.button("Confirm execute WRITE"):
                            try:
                                rows = run_write_query(proposed_sql, tuple(params))
                                st.success(f"Executed. Rows affected: {rows}")
                                append_log({"event":"write_exec","rows":rows,"sql":proposed_sql})
                            except Exception as e:
                                st.error(f"DB error: {e}")
                                append_log({"event":"db_error","error":str(e),"sql":proposed_sql})
                else:
                    # SELECT flow
                    try:
                        df = run_select_query(proposed_sql, tuple(params))
                        st.markdown("**Results (top rows):**")
                        st.dataframe(df.head(50))
                        st.markdown("**Text Preview:**")
                        st.markdown(format_df(df))
                        append_log({"event":"select_exec","rows_returned":int(len(df)),"sql":proposed_sql})
                    except Exception as e:
                        st.error(f"DB error: {e}")
                        append_log({"event":"db_error","error":str(e),"sql":proposed_sql})
with col2:
    st.markdown("### Quick actions")
    if st.button("List tables"):
        # small helper to query DB metadata
        try:
            tbl_sql = "SELECT tablename FROM iitables"  # NOTE: actual Ingres table metadata query may differ
            df = run_select_query(tbl_sql)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not list tables: {e}")

    if st.button("Show schema of table (prompt)"):
        tname = st.text_input("Table name for schema", value="")
        if tname:
            try:
                schema_sql = f"SELECT * FROM {tname} LIMIT 1"
                df = run_select_query(schema_sql)
                st.write("Columns:", list(df.columns))
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Examples**")
    st.write("- Show students with GPA > 8")
    st.write("- Count orders for last month by product")
    st.write("- Update student address where id=123 (admin only)")
    st.write("- Insert row into attendance (admin only)")

st.markdown("---")
st.markdown("Built-in safety: read-only mode by default; admin required for writes. Logs are server-side only.")
# ingres_assistant.py
import os
import time
import json
import random
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import pyodbc
import google.generativeai as genai

# -------------------------
# Config & environment
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
INGRES_ODBC_CONN = os.getenv("INGRES_ODBC_CONN", "")  # e.g. "DSN=ingres;UID=user;PWD=pass"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")  # used for admin actions in UI
READONLY_BY_DEFAULT = os.getenv("READONLY_BY_DEFAULT", "1") == "1"

if not GEMINI_API_KEY:
    st.error("Server misconfigured: GEMINI_API_KEY not found in environment.")
    st.stop()

if not INGRES_ODBC_CONN:
    st.warning("No INGRES_ODBC_CONN found - DB operations disabled. Set env var for database access if available.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# -------------------------
# Logging
# -------------------------
LOG_FILE = "assistant_logs.jsonl"
logger = logging.getLogger("ingres_assistant")
logger.setLevel(logging.INFO)

# Helper to append logs (structured)
def append_log(entry: Dict[str, Any]):
    entry["ts"] = datetime.utcnow().isoformat()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.exception("Could not write log: %s", e)

# -------------------------
# DB Interaction module
# -------------------------
def get_db_conn():
    if not INGRES_ODBC_CONN:
        raise RuntimeError("No INGRES_ODBC_CONN provided")
    return pyodbc.connect(INGRES_ODBC_CONN, autocommit=False)

def run_select_query(sql: str, params: Tuple = ()):
    """Execute a SELECT and return DataFrame."""
    conn = get_db_conn()
    try:
        df = pd.read_sql(sql, conn, params=list(params))
        return df
    finally:
        conn.close()

def run_write_query(sql: str, params: Tuple = ()):
    """Execute INSERT/UPDATE/DELETE, return rows affected."""
    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rowcount = cur.rowcount
        conn.commit()
        return rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# -------------------------
# Simple SQL safety validator
# -------------------------
DISALLOWED_TOKENS = ["DROP", "TRUNCATE", "ALTER", "EXEC", "CREATE", "GRANT", "REVOKE"]
WRITE_TOKENS = ["INSERT", "UPDATE", "DELETE"]

def validate_sql(sql: str, allow_write: bool = False) -> Tuple[bool, str]:
    """
    Returns (ok, message). Prevents dangerous statements.
    """
    upper = sql.upper()
    for tok in DISALLOWED_TOKENS:
        if tok in upper:
            return False, f"Blocked token detected: {tok}"
    if any(w in upper for w in WRITE_TOKENS) and not allow_write:
        return False, "Write operations (INSERT/UPDATE/DELETE) are not allowed in read-only mode."
    return True, "OK"

# -------------------------
# NLP Intent module (Gemini)
# -------------------------
RETRY_COUNT = 3
RETRY_SLEEP = 1.0

def classify_intent_and_sql(nl: str, system_context: str = "") -> Dict[str, Any]:
    """
    Use Gemini to: classify intent (retrieve/insert/update/delete/admin), propose a parameterized SQL,
    and return extracted parameters as JSON.
    We keep a strict prompt to ask for a JSON response: {"intent": "...", "sql": "...", "params": {...}}
    """
    prompt = f"""
You are an intelligent assistant that translates plain English into safe, parameterized SQL for an Ingres database.
Return ONLY a JSON object with exactly three fields:
- "intent": one of ["retrieve","insert","update","delete","admin","other"]
- "sql": the parameterized SQL using ? placeholders for values (use LIMIT 100 for selects by default)
- "params": an ordered list of parameter values (strings/numbers) matching the placeholders, or [] if none.

Rules:
* Do NOT include any disallowed operations (DROP, TRUNCATE, ALTER, CREATE, GRANT, REVOKE).
* For SELECT queries, prefer explicit column names if user mentions them; else use SELECT * with LIMIT 100.
* For ambiguous destructive queries, return intent "other" and an empty sql and params.
* Keep JSON minimal and valid.

Context:
{system_context}

User request:
\"\"\"{nl}\"\"\"
"""
    # call gemini
    for attempt in range(RETRY_COUNT):
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            session = model.start_chat(history=[])
            resp = session.send_message(prompt)
            text = resp.text.strip()
            # Try to parse JSON from response
            # Some model responses may include extra text — attempt to find the JSON substring
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                json_str = text[start:end]
            else:
                json_str = text
            parsed = json.loads(json_str)
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP * (attempt + 1))
    # fallback
    raise last_err

# -------------------------
# Response generation helper
# -------------------------
def format_df(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows returned."
    # Show top 10 rows
    return df.head(10).to_markdown(index=False)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Ingres Virtual Assistant", layout="wide")
st.title("Ingres Virtual Assistant — NL → SQL")

# sidebar: admin toggles (admin secret required)
with st.sidebar:
    st.header("Admin")
    admin_key = st.text_input("Admin secret (hidden)", type="password")
    is_admin = (admin_key != "" and admin_key == ADMIN_SECRET)
    if is_admin:
        st.success("Admin mode ON")
    else:
        if admin_key:
            st.error("Admin secret incorrect")
    read_only_mode = st.checkbox("Read-only mode", value=READONLY_BY_DEFAULT)
    st.markdown("---")
    st.markdown("Logs:")
    if st.button("Download logs"):
        try:
            with open(LOG_FILE, "rb") as f:
                st.download_button("Download logs file", f, file_name="assistant_logs.jsonl")
        except Exception:
            st.error("No logs available yet.")

# main chat area
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Ask the database (plain English)", height=150, placeholder="e.g. Show me students with GPA > 8.5")
    if st.button("Submit"):
        if not user_input.strip():
            st.warning("Write something please.")
        else:
            # log user query (do not log raw sensitive data in production)
            append_log({"event": "user_query", "text": user_input[:1000]})
            with st.spinner("Interpreting your request..."):
                try:
                    parsed = classify_intent_and_sql(user_input)
                except Exception as e:
                    st.error(f"Could not parse the request: {e}")
                    append_log({"event": "nl_error", "error": str(e)})
                    parsed = {"intent":"other","sql":"","params":[]}

            intent = parsed.get("intent", "other")
            proposed_sql = parsed.get("sql", "")
            params = parsed.get("params", [])

            # Validate SQL
            allow_write = is_admin and not read_only_mode
            ok, msg = validate_sql(proposed_sql, allow_write=allow_write)
            if not ok:
                st.warning(f"Query blocked: {msg}")
                append_log({"event":"blocked_query","reason":msg,"intent":intent,"sql":proposed_sql})
            else:
                st.markdown("**Proposed query:**")
                st.code(proposed_sql)
                st.write("Parameters:", params)
                # Confirm before writes if not admin
                if any(tok in proposed_sql.upper() for tok in WRITE_TOKENS):
                    if not is_admin:
                        st.error("Write operations require admin privileges.")
                        append_log({"event":"write_blocked","sql":proposed_sql})
                    else:
                        if st.button("Confirm execute WRITE"):
                            try:
                                rows = run_write_query(proposed_sql, tuple(params))
                                st.success(f"Executed. Rows affected: {rows}")
                                append_log({"event":"write_exec","rows":rows,"sql":proposed_sql})
                            except Exception as e:
                                st.error(f"DB error: {e}")
                                append_log({"event":"db_error","error":str(e),"sql":proposed_sql})
                else:
                    # SELECT flow
                    try:
                        df = run_select_query(proposed_sql, tuple(params))
                        st.markdown("**Results (top rows):**")
                        st.dataframe(df.head(50))
                        st.markdown("**Text Preview:**")
                        st.markdown(format_df(df))
                        append_log({"event":"select_exec","rows_returned":int(len(df)),"sql":proposed_sql})
                    except Exception as e:
                        st.error(f"DB error: {e}")
                        append_log({"event":"db_error","error":str(e),"sql":proposed_sql})
with col2:
    st.markdown("### Quick actions")
    if st.button("List tables"):
        # small helper to query DB metadata
        try:
            tbl_sql = "SELECT tablename FROM iitables"  # NOTE: actual Ingres table metadata query may differ
            df = run_select_query(tbl_sql)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not list tables: {e}")

    if st.button("Show schema of table (prompt)"):
        tname = st.text_input("Table name for schema", value="")
        if tname:
            try:
                schema_sql = f"SELECT * FROM {tname} LIMIT 1"
                df = run_select_query(schema_sql)
                st.write("Columns:", list(df.columns))
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Examples**")
    st.write("- Show students with GPA > 8")
    st.write("- Count orders for last month by product")
    st.write("- Update student address where id=123 (admin only)")
    st.write("- Insert row into attendance (admin only)")

st.markdown("---")
st.markdown("Built-in safety: read-only mode by default; admin required for writes. Logs are server-side only.")
