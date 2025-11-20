import os
import io
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
from streamlit_folium import st_folium

# Optional LLM (Gemini) integration — only used if available and API key set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HAS_GENAI = False
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

# Working dirs & logs
WORKDIR = Path.cwd()
STATE_DIR = WORKDIR / "ingres_state"
STATE_DIR.mkdir(exist_ok=True)
LOG_FILE = STATE_DIR / "session_logs.json"
if not LOG_FILE.exists():
    LOG_FILE.write_text("[]", encoding="utf-8")


def log_event(user: str, event: str, note: str = ""):
    try:
        logs = json.loads(LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        logs = []
    logs.append({"ts": datetime.utcnow().isoformat() + "Z", "user": user, "event": event, "note": note})
    LOG_FILE.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------
# Demo district-level data (2020-2024)
# ------------------------------

def load_demo_data():
    districts = [
        ("Karnataka", "Bengaluru Urban", 12.9716, 77.5946, [23.1, 24.3, 25.7, 26.8, 27.2], [18, 19, 20, 21, 22]),
        ("Karnataka", "Mysuru", 12.2958, 76.6394, [17.2, 18.1, 18.9, 19.6, 20.4], [27, 28, 29, 30, 31]),
        ("Karnataka", "Belagavi", 15.8497, 74.4977, [13.2, 14.1, 15.3, 16.4, 17.2], [35, 36, 38, 40, 41]),
        ("Maharashtra", "Mumbai", 19.0760, 72.8777, [14.5, 15.1, 15.8, 16.3, 16.9], [42, 44, 46, 48, 50]),
        ("Maharashtra", "Pune", 18.5204, 73.8567, [21.2, 22.1, 23.3, 24.4, 25.2], [60, 62, 64, 66, 68]),
        ("Maharashtra", "Nagpur", 21.1458, 79.0882, [28.1, 29.3, 30.7, 31.9, 33.1], [75, 78, 82, 86, 90]),
    ]
    years = [2020, 2021, 2022, 2023, 2024]
    rows = []
    for state, dist, lat, lon, gwl_list, res_list in districts:
        for i, y in enumerate(years):
            rows.append({
                "State": state,
                "District": dist,
                "Latitude": float(lat),
                "Longitude": float(lon),
                "Year": int(y),
                "Avg_GWL_m": float(gwl_list[i]),
                "Resources_TMC": float(res_list[i]),
            })
    return pd.DataFrame(rows)

DF_DEMO = load_demo_data()


# ------------------------------
# Utilities
# ------------------------------

def normalize_df(d: pd.DataFrame, yr_range) -> pd.DataFrame:
    dfc = d.copy()
    colmap = {}
    for c in dfc.columns:
        lc = c.lower()
        if lc in ("state", "state_name"):
            colmap[c] = "State"
        if lc in ("district", "district_name"):
            colmap[c] = "District"
        if lc in ("lat", "latitude", "y"):
            colmap[c] = "Latitude"
        if lc in ("lon", "lng", "longitude", "x"):
            colmap[c] = "Longitude"
        if "ground" in lc or "gwl" in lc or "avg_gwl" in lc or "waterlevel" in lc or "level" in lc:
            colmap[c] = "Avg_GWL_m"
        if "resource" in lc or "tmc" in lc:
            colmap[c] = "Resources_TMC"
        if "year" in lc:
            colmap[c] = "Year"
    if colmap:
        try:
            dfc = dfc.rename(columns=colmap)
        except Exception:
            pass
    if "State" not in dfc.columns:
        dfc["State"] = "Unknown"
    if "District" not in dfc.columns:
        dfc["District"] = dfc.get("District", dfc["State"])
    if "Year" not in dfc.columns:
        dfc["Year"] = yr_range[0]
    if "Avg_GWL_m" not in dfc.columns:
        numeric_cols = [c for c in dfc.select_dtypes(include=[np.number]).columns if c.lower() not in ("latitude","longitude","year")]
        if numeric_cols:
            dfc = dfc.rename(columns={numeric_cols[0]: "Avg_GWL_m"})
        else:
            dfc["Avg_GWL_m"] = np.nan
    if "Latitude" not in dfc.columns or "Longitude" not in dfc.columns:
        coords = []
        for _, r in dfc.iterrows():
            lat = r.get("Latitude")
            lon = r.get("Longitude")
            if pd.notna(lat) and pd.notna(lon):
                coords.append((float(lat), float(lon)))
            else:
                match = DF_DEMO[(DF_DEMO["State"] == r.get("State")) & (DF_DEMO["District"] == r.get("District"))]
                if not match.empty:
                    coords.append((float(match.iloc[0]["Latitude"]), float(match.iloc[0]["Longitude"])))
                else:
                    match2 = DF_DEMO[DF_DEMO["State"] == r.get("State")]
                    if not match2.empty:
                        coords.append((float(match2["Latitude"].mean()), float(match2["Longitude"].mean())))
                    else:
                        coords.append((17.5, 76.0))
        dfc["Latitude"] = [c[0] for c in coords]
        dfc["Longitude"] = [c[1] for c in coords]
    try:
        dfc["Year"] = dfc["Year"].astype(int)
    except Exception:
        pass
    for col in ("Latitude", "Longitude", "Avg_GWL_m", "Resources_TMC"):
        if col in dfc.columns:
            try:
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            except Exception:
                pass
    return dfc


# LLM wrapper (defensive)

def ask_llm(prompt: str, max_tokens: int = 512) -> str:
    if not HAS_GENAI or not GEMINI_API_KEY:
        return "[LLM unavailable — set GEMINI_API_KEY or use rule-based chatbot]"
    try:
        if hasattr(genai, "generate"):
            resp = genai.generate(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), prompt=prompt, max_output_tokens=max_tokens)
            if isinstance(resp, dict):
                cand = resp.get("candidates") or resp.get("outputs")
                if cand and isinstance(cand, list):
                    return str(cand[0].get("content") or cand[0].get("text") or cand[0])
                return str(resp)
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)
        if hasattr(genai, "generate_text"):
            resp = genai.generate_text(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), prompt=prompt, max_output_tokens=max_tokens)
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)
        return "[LLM: unsupported google.generativeai API surface]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="IN-GRES — Groundwater Explorer", layout="wide")
st.title("🌊 IN-GRES — District Groundwater Explorer (Demo)")

# Theme option: Light / Dark
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    st.header("Controls")
    theme_choice = st.radio("Theme", ["Light", "Dark"], index=0)
    st.session_state.theme = theme_choice
    data_source = st.selectbox("Data source", ["Demo dataset", "Upload CSV/XLSX"]) 
    st.markdown("---")
    st.markdown("### Data options")
    uploaded = st.file_uploader("Upload sample CSV or Excel", type=["csv", "xlsx", "xls"], key="uploader")
    if st.button("Load demo sample data"):
        st.session_state.uploaded_df = None
        st.session_state.loaded_demo = True
        st.experimental_rerun()

    st.markdown("\n---\nOptional: enable LLM for chatbot (requires GEMINI_API_KEY env var)")
    st.write(f"LLM enabled: {'Yes' if HAS_GENAI and GEMINI_API_KEY else 'No'}")

# Apply simple CSS for theme
if st.session_state.theme == "Dark":
    st.markdown(
        """
        <style>
        .main {background-color: #0e1117; color: #e6eef8}
        .stButton>button {background-color:#2b3038}
        .css-1d391kg {background-color:#0e1117}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("""<style> .main {background-color:unset} </style>""", unsafe_allow_html=True)

# Load dataframe (Demo or uploaded)
if data_source == "Demo dataset" or (hasattr(st.session_state, "loaded_demo") and st.session_state.loaded_demo):
    df = DF_DEMO.copy()
else:
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.uploaded_df = df.copy()
        except Exception as e:
            st.error(f"Failed to read upload: {e}")
            df = DF_DEMO.copy()
    else:
        # no upload provided
        df = DF_DEMO.copy()

# allow user to replace with session uploaded
if hasattr(st.session_state, "uploaded_df") and st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df.copy()

# Controls for filtering
states = st.multiselect("States", df["State"].unique().tolist(), default=df["State"].unique().tolist())
year_min = int(df["Year"].min())
year_max = int(df["Year"].max())
year_range = st.slider("Year range", year_min, year_max, (year_min, year_max))
show_points = st.checkbox("Show map points", True)
show_heatmap = st.checkbox("Show heatmap", True)
show_tl = st.checkbox("Show time-lapse (demo)", True)
ask_ai = st.checkbox("Ask LLM for explanation (chatbot)", False)

# Normalize and filter
try:
    df = normalize_df(df, year_range)
except Exception:
    df = DF_DEMO.copy()

if states:
    df = df[df["State"].isin(states)].copy()

df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])].copy()

if df.empty:
    st.warning("No data after filtering — showing demo subset for selected states/years")
    df = DF_DEMO[(DF_DEMO["State"].isin(states)) & (DF_DEMO["Year"].between(year_range[0], year_range[1]))].copy()

# Show preview
st.markdown("### Data preview")
st.dataframe(df.reset_index(drop=True).head(100))

# After sample data table: put everything into a dropdown/expander
with st.expander("Show dashboard (charts, maps, exports, forecast, time-lapse, TTS, AI explanation)", expanded=False):
    st.markdown("## Maps")
    map_center = [float(df["Latitude"].mean()), float(df["Longitude"].mean())]
    map_tiles = "CartoDB dark_matter" if st.session_state.theme == "Dark" else "CartoDB positron"
    m = folium.Map(location=map_center, zoom_start=6, tiles=map_tiles)

    if show_heatmap:
        heat_rows = df[["Latitude", "Longitude", "Avg_GWL_m"]].dropna().values.tolist()
        if heat_rows:
            try:
                HeatMap(heat_rows, radius=15, blur=18).add_to(m)
            except Exception:
                pass

    if show_points:
        for _, r in df.dropna(subset=["Latitude", "Longitude"]).iterrows():
            try:
                popup_html = f"<b>{r.get('District')}</b><br/>{r.get('State')}<br/>Year: {int(r.get('Year'))}<br/>GWL: {r.get('Avg_GWL_m')}"
            except Exception:
                popup_html = f"{r.get('District')} - {r.get('State')}"
            popup = folium.Popup(html=popup_html, max_width=250)
            folium.CircleMarker(location=[r["Latitude"], r["Longitude"]], radius=6, color="#00bfff", fill=True, fill_opacity=0.8, popup=popup).add_to(m)

    st_folium(m, width=1100, height=450)

    # Time-lapse
    if show_tl:
        st.markdown("### Time-lapse (demo dataset)")
        features = []
        demo_subset = DF_DEMO[(DF_DEMO["Year"] >= year_range[0]) & (DF_DEMO["Year"] <= year_range[1])]
        for _, r in demo_subset.iterrows():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [r["Longitude"], r["Latitude"]]},
                "properties": {"time": f"{int(r['Year'])}-01-01", "popup": f"{r['District']} ({r['State']}) — {r['Avg_GWL_m']} m"}
            })
        tm = folium.Map(location=map_center, zoom_start=6, tiles=map_tiles)
        try:
            TimestampedGeoJson({"type": "FeatureCollection", "features": features}, period="P1Y", duration="P1Y", auto_play=False, loop=False, add_last_point=True).add_to(tm)
            st_folium(tm, width=1100, height=450)
        except Exception as e:
            st.info("Browser may block timestamped animation; try a simpler map if rendering fails.")
            st.write(str(e))

    # Charts
    st.markdown("## Charts")
    col1, col2 = st.columns(2)
    with col1:
        try:
            fig_line = px.line(df, x="Year", y="Avg_GWL_m", color="District", markers=True, title="Groundwater trend by district")
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception as e:
            st.warning("Line chart rendering failed: " + str(e))

    with col2:
        try:
            fig_hist = px.histogram(df, x="Avg_GWL_m", color="State", nbins=12, title="Distribution of groundwater levels")
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.warning("Histogram rendering failed: " + str(e))

    try:
        st.plotly_chart(px.box(df, x="Year", y="Avg_GWL_m", points="all", title="Year-wise distribution"), use_container_width=True)
        st.plotly_chart(px.bar(df, x="District", y="Resources_TMC", color="Year", title="Resources (TMC) by district/year"), use_container_width=True)
    except Exception:
        pass

    # Delta map
    st.markdown("### Change / Delta Map")
    y0, y1 = year_range
    try:
        tmp = df.copy()
        tmp["_lat_r"] = tmp["Latitude"].round(3)
        tmp["_lon_r"] = tmp["Longitude"].round(3)
        agg = tmp.groupby(["_lat_r", "_lon_r", "Year"]) ["Avg_GWL_m"].median().reset_index()
        pivot = agg.pivot_table(index=["_lat_r", "_lon_r"], columns="Year", values="Avg_GWL_m")
        if y0 in pivot.columns and y1 in pivot.columns:
            pivot["delta"] = pivot[y1] - pivot[y0]
            pc = pivot.reset_index().dropna(subset=["delta"])
            if not pc.empty:
                fig_delta = px.scatter_mapbox(pc, lat="_lat_r", lon="_lon_r", color="delta", size=pc["delta"].abs(), color_continuous_scale="RdYlBu", zoom=6, title=f"Change {y1} - {y0}")
                fig_delta.update_layout(mapbox_style="carto-positron")
                st.plotly_chart(fig_delta, use_container_width=True)
            else:
                st.info("No overlapping rounded coordinates to compute deltas.")
        else:
            st.info("Need data for both range endpoints to compute delta.")
    except Exception as e:
        st.warning("Delta map failed: " + str(e))

    # Forecasting
    st.markdown("## Forecasting (simple linear)")
    forecast_horizon = st.slider("Forecast horizon (years ahead)", 1, 10, 5, key="forecast_horizon")
    fc_rows = []
    for d in DF_DEMO["District"].unique():
        tmp = DF_DEMO[DF_DEMO["District"] == d]
        if tmp.shape[0] >= 2:
            coef = np.polyfit(tmp["Year"].values, tmp["Avg_GWL_m"].values, 1)
            poly = np.poly1d(coef)
            last_year = int(tmp["Year"].max())
            for i in range(1, forecast_horizon + 1):
                yr = last_year + i
                fc_rows.append({"District": d, "Year": yr, "Forecast_GWL": float(poly(yr))})
    if fc_rows:
        fc_df = pd.DataFrame(fc_rows)
        st.plotly_chart(px.line(fc_df, x="Year", y="Forecast_GWL", color="District", title="Linear forecast"), use_container_width=True)
    else:
        st.info("Not enough demo data to forecast.")

    # Exports
    st.markdown("## Exports")
    summary_text = f"IN-GRES summary for states {', '.join(states)}. Years {year_range[0]}-{year_range[1]}. Generated at {datetime.utcnow().isoformat()}Z"
    try:
        # simple CSV download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="ingres_filtered.csv", mime="text/csv")
    except Exception as e:
        st.error("CSV export failed: " + str(e))

    # TTS: minimal (only if gTTS installed)
    try:
        from gtts import gTTS
        HAS_GTTS = True
    except Exception:
        HAS_GTTS = False

    st.markdown("## Text-to-Speech (optional)")
    tts_text = st.text_area("Text to speak", value=summary_text)
    if st.button("Generate TTS audio"):
        if HAS_GTTS:
            try:
                tts = gTTS(text=tts_text, lang="en")
                bio = io.BytesIO()
                tts.write_to_fp(bio)
                bio.seek(0)
                st.audio(bio)
                st.download_button("Download TTS (mp3)", data=bio.getvalue(), file_name="ingres_tts.mp3", mime="audio/mpeg")
            except Exception as e:
                st.error("gTTS failed: " + str(e))
        else:
            st.info("No gTTS installed in environment.")

    # Session logs viewer
    if st.checkbox("Show session log (this machine)"):
        try:
            logs = json.loads(LOG_FILE.read_text(encoding="utf-8"))
            st.write(logs[-100:])
        except Exception:
            st.write("No logs available.")

# ------------------------------
# Simple chatbot (explain data)
# ------------------------------
st.markdown("---")
st.markdown("## Chatbot — ask about the loaded data")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_col1, chat_col2 = st.columns([3, 1])
with chat_col1:
    user_input = st.text_input("Ask the chatbot to explain the data (examples: 'summarize', 'which district has highest gwl', 'trend in Karnataka')")
with chat_col2:
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("user", user_input))

# generate response if new message
if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
    last_q = st.session_state.chat_history[-1][1]
    # If LLM allowed and ask_ai toggle true, use LLM
    if ask_ai and HAS_GENAI and GEMINI_API_KEY:
        prompt = "You are a helpful assistant. " + last_q + "\n\nData snapshot:\n" + df.head(20).to_csv(index=False)
        resp = ask_llm(prompt)
    else:
        # Rule-based responder
        q = last_q.lower()
        resp = "I couldn't find an answer. Try asking for 'summary', 'highest', 'lowest', or 'trend in <State>'."
        try:
            if "summary" in q or "summar" in q:
                rows = len(df)
                years = sorted(df["Year"].unique())
                resp = f"Data summary: {rows} rows, years {years[0]}-{years[-1]}. States: {', '.join(sorted(df['State'].unique()))}."
            elif "highest" in q or "max" in q:
                col = "Avg_GWL_m"
                if col in df.columns:
                    r = df.loc[df[col].idxmax()]
                    resp = f"Highest {col}: {r[col]} m at {r['District']}, {r['State']} (Year {int(r['Year'])})."
            elif "lowest" in q or "min" in q:
                col = "Avg_GWL_m"
                if col in df.columns:
                    r = df.loc[df[col].idxmin()]
                    resp = f"Lowest {col}: {r[col]} m at {r['District']}, {r['State']} (Year {int(r['Year'])})."
            elif "trend" in q or "change" in q:
                # try to parse state
                matched_state = None
                for s in df['State'].unique():
                    if s.lower() in q:
                        matched_state = s
                        break
                if matched_state:
                    tmp = df[df['State'] == matched_state]
                    if 'Year' in tmp.columns and 'Avg_GWL_m' in tmp.columns:
                        byyr = tmp.groupby('Year')['Avg_GWL_m'].median().reset_index()
                        if byyr.shape[0] >= 2:
                            coef = np.polyfit(byyr['Year'], byyr['Avg_GWL_m'], 1)
                            slope = coef[0]
                            trend = 'increasing depth (depletion)' if slope > 0 else 'decreasing depth (recovery)' if slope < 0 else 'stable'
                            resp = f"Trend for {matched_state}: median groundwater level shows {trend} (slope {slope:.3f} m/year)."
                        else:
                            resp = f"Not enough yearly data for {matched_state} to compute trend."
                else:
                    resp = "Specify a state (e.g., 'trend in Karnataka') to get a state-level trend."
            else:
                resp = "Try commands like: 'summary', 'which district has highest gwl', 'lowest gwl', or 'trend in <State>'."
        except Exception as e:
            resp = f"Error while processing: {e}"

    st.session_state.chat_history.append(("bot", resp))

# Display chat history
for who, text in st.session_state.chat_history:
    if who == 'user':
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

# Final log
user_tag = os.getenv('USER', 'local_user')
log_event(user_tag, 'view_dashboard', f"states={states}, years={year_range}")

# End of file
