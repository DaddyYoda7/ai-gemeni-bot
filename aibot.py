# aibot.py
import os
import time
import json
import logging
from datetime import datetime
from typing import Tuple, Optional, Dict, Any, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # Continue without dotenv if not installed

import streamlit as st
import pandas as pd
import pyodbc
import google.generativeai as genai

# ==============================================================================
# 1. Configuration and Setup
# ==============================================================================

# --- Environment Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
INGRES_ODBC_CONN = os.getenv("INGRES_ODBC_CONN", "") 
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "") 
READONLY_BY_DEFAULT = os.getenv("READONLY_BY_DEFAULT", "1") == "1"
MODEL_NAME = "gemini-2.0-flash"

# --- Constants ---
LOG_FILE = "assistant_logs.jsonl"
DISALLOWED_TOKENS = ["DROP", "TRUNCATE", "ALTER", "EXEC", "CREATE", "GRANT", "REVOKE"]
WRITE_TOKENS = ["INSERT", "UPDATE", "DELETE"]
RETRY_COUNT = 3
RETRY_SLEEP = 1.0
TARGET_TABLES = ['employees', 'departments'] # Tables the assistant should know about

# Initial checks
if not GEMINI_API_KEY:
    st.error("Server misconfigured: GEMINI_API_KEY not found. Please check your .env file.")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)


# ==============================================================================
# 2. Styling and Utilities (New GUI Enhancements)
# ==============================================================================

def apply_custom_css():
    """Applies custom CSS for a cleaner, modern look."""
    st.markdown("""
        <style>
        .stButton>button {
            border-radius: 8px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
        }
        .main-header {
            font-size: 2.5em;
            font-weight: 700;
            color: #1a73e8; /* Google Blue */
        }
        .stProgress > div > div > div > div {
            background-color: #1a73e8;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Custom card for status */
        .status-card {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .status-card.success { background-color: #e6ffed; border-left: 5px solid #00a854; }
        .status-card.warning { background-color: #fffbe6; border-left: 5px solid #ffcc00; }
        .status-card.error { background-color: #ffe6e6; border-left: 5px solid #cc0000; }
        </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# 3. Log Manager
# ==============================================================================

class LogManager:
    """Handles structured logging for all assistant actions."""
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logger = logging.getLogger("ingres_assistant")
        self.logger.setLevel(logging.INFO)

    def append_log(self, event: str, **data: Any):
        """Appends a structured log entry."""
        entry = {"event": event, **data}
        entry["ts"] = datetime.utcnow().isoformat()
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.exception("Could not write log: %s", e)

# ==============================================================================
# 4. Database Assistant (Enhanced with Schema Retrieval)
# ==============================================================================

class DatabaseAssistant:
    """Manages Gemini interaction, SQL validation, and DB execution."""

    def __init__(self, odbc_conn: str, model_name: str, log_manager: LogManager):
        self.odbc_conn = odbc_conn
        self.model_name = model_name
        self.log_manager = log_manager

    # --- DB Connection Helpers ---
    def _get_db_conn(self):
        if not self.odbc_conn:
            raise RuntimeError("Database connection is disabled. Set INGRES_ODBC_CONN in your .env file.")
        # Added login timeout to prevent indefinite waiting
        return pyodbc.connect(self.odbc_conn, autocommit=False, timeout=5)

    def run_select_query(self, sql: str, params: Tuple = ()) -> pd.DataFrame:
        conn = self._get_db_conn()
        try:
            df = pd.read_sql(sql, conn, params=list(params))
            return df
        finally:
            conn.close()

    def run_write_query(self, sql: str, params: Tuple = ()) -> int:
        conn = self._get_db_conn()
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

    # --- Schema Retrieval ---
    def get_schema_context(self, tables: List[str]) -> str:
        """Queries Ingres system tables for schema details of specified tables."""
        
        table_list_str = ", ".join([f"'{t}'" for t in tables])
        
        # Use uppercase table names as Ingres is case-insensitive for unquoted identifiers
        schema_query = f"""
        SELECT c.table_name, c.column_name, c.column_datatype
        FROM iicolumns c
        WHERE UPPER(c.table_name) IN ({table_list_str.upper()})
        ORDER BY c.table_name, c.column_sequence;
        """
        
        try:
            df_schema = self.run_select_query(schema_query)
        except Exception as e:
            self.log_manager.append_log("schema_fetch_error", error=str(e))
            return f"Error fetching schema: {e}"

        if df_schema.empty:
            return f"Could not find schema information for tables: {', '.join(tables)}."

        # Format schema into a context string for the LLM
        schema_context = []
        for table_name, group in df_schema.groupby('table_name'):
            # Convert column names to lower case for consistent LLM prompting
            columns = [f"{row.column_name.lower()} ({row.column_datatype.lower()})" for row in group.itertuples()]
            schema_context.append(f"Table '{table_name.lower()}': Columns: {', '.join(columns)}")
        
        return "Database Schema Context:\n" + "\n".join(schema_context)

    # --- SQL Safety ---
    def validate_sql(self, sql: str, allow_write: bool = False) -> Tuple[bool, str]:
        """Returns (ok, message). Prevents dangerous statements."""
        upper = sql.upper()
        for tok in DISALLOWED_TOKENS:
            if tok in upper:
                return False, f"Blocked token detected: **{tok}**"
        if any(w in upper for w in WRITE_TOKENS) and not allow_write:
            return False, "Write operations are not allowed in **read-only** mode."
        return True, "OK"

    # --- Helper for robust JSON extraction ---
    def _extract_json(self, text: str) -> str:
        """
        Extracts JSON string, handling potential Markdown code fences (```json) 
        and finding the boundary of the main JSON object ({...}).
        """
        text = text.strip()
        
        # Strip common markdown code fences
        if text.startswith('```'):
            text_lines = text.split('\n')
            if len(text_lines) > 2 and text_lines[0].strip().lower() in ('```json', '```sql', '```'):
                text = "\n".join(text_lines[1:-1])
            elif text.endswith('```'):
                text = text.strip('`')

        # Find the outermost JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start != -1 and end != -1:
            return text[start:end]
        
        # If no clear JSON block is found, return the original text, letting json.loads fail
        return text

    # --- Gemini NLP Logic (Centralized) ---
    def classify_intent_and_sql(self, nl: str, system_context: str = "") -> Dict[str, Any]:
        prompt = f"""
You are an intelligent assistant that translates plain English into safe, parameterized SQL for an Ingres database.
Return ONLY a JSON object with exactly three fields, without any preceding or following text or markdown formatting:
- "intent": one of ["retrieve","insert","update","delete","admin","other"]
- "sql": the parameterized Ingres-compatible SQL using ? placeholders for values (use LIMIT 100 for selects by default)
- "params": an ordered list of parameter values (strings/numbers) matching the placeholders, or [] if none.

Rules:
* Do NOT include any disallowed operations (DROP, TRUNCATE, ALTER, CREATE, GRANT, REVOKE).
* For SELECT queries, prefer explicit column names if user mentions them; else use SELECT * with LIMIT 100.
* For ambiguous destructive queries, return intent "other" and an empty sql and params.
* Return only the raw JSON.

Context:
{system_context}

User request:
\"\"\"{nl}\"\"\"
"""
        for attempt in range(RETRY_COUNT):
            try:
                model = genai.GenerativeModel(self.model_name)
                resp = model.generate_content(prompt)
                text = resp.text.strip()
                
                # Use robust JSON extraction
                json_str = self._extract_json(text)
                
                parsed = json.loads(json_str)
                self.log_manager.append_log("nl_parse_success", nl=nl[:100], result=parsed)
                return parsed
            except Exception as e:
                last_err = e
                time.sleep(RETRY_SLEEP * (attempt + 1))
        
        self.log_manager.append_log("nl_parse_fail", nl=nl[:100], error=str(last_err))
        raise last_err

# ==============================================================================
# 5. Streamlit UI
# ==============================================================================

# Apply styling first
apply_custom_css()

# Initialize manager and assistant
log_manager = LogManager(LOG_FILE)
assistant = DatabaseAssistant(INGRES_ODBC_CONN, MODEL_NAME, log_manager)

# --- Schema Caching ---
@st.cache_resource(show_spinner="Connecting to Ingres and fetching database schema...")
def load_db_schema(_assistant, _tables):
    """Load the database schema into memory once."""
    if not INGRES_ODBC_CONN:
        return "Schema unavailable (No INGRES_ODBC_CONN set)."
    try:
        return _assistant.get_schema_context(_tables)
    except Exception as e:
        return f"Failed to fetch schema. DB Error: {e}"

schema_context = load_db_schema(assistant, TARGET_TABLES)

st.set_page_config(page_title="Ingres Virtual Assistant 📊", layout="wide")
st.markdown('<p class="main-header">Ingres Virtual Assistant — NL → SQL 🤖</p>', unsafe_allow_html=True)

# Initialize session state for consistent results display
if "results_data" not in st.session_state:
    st.session_state.results_data = {"sql": "", "params": [], "result": "Awaiting query..."}

# --- Sidebar (Configuration and Status) ---
with st.sidebar:
    st.header("⚙️ Configuration & Status")
    
    # DB Status Card
    if INGRES_ODBC_CONN:
        db_status = '<div class="status-card success">DB Status: **Connected** (ODBC OK)</div>'
    else:
        db_status = '<div class="status-card error">DB Status: **Disabled** (Missing `INGRES_ODBC_CONN`)</div>'
    st.markdown(db_status, unsafe_allow_html=True)
    
    # Read-Only Toggle
    read_only_mode = st.toggle(
        "Enable Read-only Mode", 
        value=READONLY_BY_DEFAULT,
        help="When enabled, INSERT, UPDATE, and DELETE queries will be blocked."
    )
    
    st.markdown("---")
    st.subheader("🔑 Admin Access")
    
    # Admin Key Input
    admin_key = st.text_input("Admin Secret", type="password", help="Enter the ADMIN_SECRET from your .env file to enable writes.")
    is_admin = (admin_key != "" and admin_key == ADMIN_SECRET)
    
    if is_admin:
        st.success("Admin mode **Active**! You can execute write queries.")
    elif admin_key:
        st.error("Admin secret incorrect.")

    st.markdown("---")
    st.subheader("Context Reference")
    with st.expander("Database Schema Context"):
        st.code(schema_context, language="text")

    st.subheader("Diagnostics")
    if st.button("Download Logs 💾", use_container_width=True):
        try:
            with open(LOG_FILE, "rb") as f:
                st.download_button("Download logs file", f, file_name="assistant_logs.jsonl", mime="application/json", use_container_width=True)
        except FileNotFoundError:
            st.warning("No logs available yet.")

# --- Main Query Area ---
st.markdown("## 💬 Natural Language Query")
query_col, examples_col = st.columns([3, 1])

with query_col:
    user_input = st.text_area(
        "Translate your question into Ingres SQL:", 
        height=100, 
        placeholder="e.g. Find the highest paid employee in Boston and their job title."
    )
    submit_button = st.button("Generate & Execute SQL 🚀", use_container_width=True, type="primary")

with examples_col:
    st.caption("Common Queries")
    st.info("• Salaries in Research")
    st.info("• Count new hires in 2024")
    st.info("• Update Jane Doe's salary (Admin)")

# --- Execution Logic ---
if submit_button and user_input.strip():
    log_manager.append_log("user_query", text=user_input[:1000])
    
    with st.spinner("Processing request via Gemini..."):
        try:
            parsed = assistant.classify_intent_and_sql(user_input, system_context=schema_context)
        except Exception as e:
            st.error(f"NLP Error: Could not parse the request into JSON: {e}")
            log_manager.append_log("nl_error", error=str(e))
            st.session_state.results_data = {"sql": "", "params": [], "result": "NLP failed to generate valid structured response."}
            parsed = {"intent":"other","sql":"","params":[]}

    if parsed.get("sql"):
        intent = parsed.get("intent", "other")
        proposed_sql = parsed.get("sql", "")
        params = parsed.get("params", [])
        
        allow_write = is_admin and not read_only_mode
        ok, msg = assistant.validate_sql(proposed_sql, allow_write=allow_write)
        
        st.session_state.results_data["sql"] = proposed_sql
        st.session_state.results_data["params"] = params

        if not ok:
            st.warning(f"🚫 **Query Blocked:** {msg}")
            log_manager.append_log("blocked_query", reason=msg, intent=intent, sql=proposed_sql)
            st.session_state.results_data["result"] = f"Query Blocked: {msg}"
        
        elif intent in ["insert", "update", "delete"]:
            if not allow_write:
                st.error("🔒 **Write Blocked:** Requires Admin Access and Read-only mode OFF.")
                st.session_state.results_data["result"] = "Write blocked: Admin or read-only mode required."
                log_manager.append_log("write_blocked", sql=proposed_sql)
            else:
                # Stage write operation for confirmation
                st.session_state.execute_write_sql = proposed_sql
                st.session_state.execute_write_params = params
                st.session_state.results_data["result"] = "Awaiting Write Confirmation."
        else: # SELECT flow
            try:
                with st.spinner("Executing SELECT query against Ingres..."):
                    df = assistant.run_select_query(proposed_sql, tuple(params))
                    st.session_state.results_data["df"] = df
                    st.session_state.results_data["result"] = f"✅ Success: Query executed. Returned **{len(df)}** rows."
                    log_manager.append_log("select_exec", rows_returned=int(len(df)), sql=proposed_sql)
            except Exception as e:
                st.error(f"❌ **DB Execution Error:** {e}")
                st.session_state.results_data["result"] = f"DB Error: {e}"
                log_manager.append_log("db_error", error=str(e), sql=proposed_sql)
    else:
        st.error("The assistant could not formulate a valid SQL query.")
        st.session_state.results_data["result"] = "NLP failed to generate SQL."
        
elif submit_button and not user_input.strip():
    st.warning("Please enter a query.")

# --- Results and Confirmation Area ---
st.markdown("---")
st.markdown("## 📋 Execution Results")

# Display the proposed query details in a structured panel
col1, col2 = st.columns([1, 4])

with col1:
    st.markdown(f"**Status:**")
    st.success(st.session_state.results_data['result'].split(':')[0]) if "Success" in st.session_state.results_data['result'] else st.warning(st.session_state.results_data['result'].split(':')[0])

with col2:
    st.code(st.session_state.results_data.get("sql", "N/A"), language="sql", line_numbers=True)
    if st.session_state.results_data.get('params'):
        st.caption(f"Parameters: {st.session_state.results_data.get('params')}")

# Handle Write Confirmation Separately
if 'execute_write_sql' in st.session_state and st.session_state.execute_write_sql:
    st.markdown("### ✍️ Confirm Write Operation")
    st.error("⚠️ **CRITICAL ACTION:** You are about to execute a WRITE query. Please review the SQL carefully.")
    
    confirm_col, cancel_col = st.columns([1, 1])
    
    with confirm_col:
        if st.button("Confirm Execute WRITE 🛡️", key="confirm_write_btn", type="primary", use_container_width=True):
            try:
                rows = assistant.run_write_query(
                    st.session_state.execute_write_sql, 
                    tuple(st.session_state.execute_write_params)
                )
                st.session_state.results_data["result"] = f"🎉 Success: Executed WRITE. Rows affected: **{rows}**."
                log_manager.append_log("write_exec", rows=rows, sql=st.session_state.execute_write_sql)
            except Exception as e:
                st.error(f"❌ DB error during WRITE: {e}")
                log_manager.append_log("db_error", error=str(e), sql=st.session_state.execute_write_sql)
                st.session_state.results_data["result"] = f"DB Error during Write: {e}"
            
            # Clear the write state regardless of success/fail
            del st.session_state.execute_write_sql
            del st.session_state.execute_write_params
            st.rerun()
            
    with cancel_col:
        if st.button("Cancel Write", key="cancel_write_btn", use_container_width=True):
            st.session_state.results_data["result"] = "Write operation canceled by user."
            del st.session_state.execute_write_sql
            del st.session_state.execute_write_params
            st.rerun()

# Display DataFrame if available
if "df" in st.session_state.results_data and not st.session_state.results_data["df"].empty:
    st.markdown("### 📊 Data Preview")
    st.dataframe(st.session_state.results_data["df"], use_container_width=True)