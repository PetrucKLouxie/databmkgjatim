import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import streamlit.components.v1 as components
import base64
import requests

# --- 1. CONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="BMKG Jatim Weather Intelligence",
    page_icon="📡",
    layout="wide"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #1c252e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
    }
    [data-testid="stMetricValue"] { color: #58a6ff; font-weight: bold; font-size: 24px; }
    [data-testid="stMetricLabel"] { color: #8b949e; font-size: 14px; text-transform: uppercase; }
    h1, h2, h3 { color: #f0f6fc; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DAFTAR STASIUN ---
STASIUN_LIST = [
    "Stageof Tretes", "Stamet Kalianget", "Stamet Tuban", "Staklim Malang",
    "Stamet Banyuwangi", "Stamet Juanda", "Stamet Bawean", "Stageof Karang Kates",
    "Stamet Perak 1", "Stamar Perak 2", "Stageof Nganjuk"
]
# ==========================
# REQUIRED COLUMN VALIDATION
# ==========================
REQUIRED_COLUMNS = [
    "Tanggal",
    "T '07.00",
    "T '13.00",
    "T '18.00",
    "TRata2",
    "TMax",
    "TMin",
    "Curah Hujan (mm)",
    "SS (%)",
    "Tekanan Udara (mb)",
    "RH07.00",
    "RH13.00",
    "RH18.00",
    "RHRata2",
    "Kec Rata2",
    "Arah Terbanyak",
    "Kec,Max",
    "Arah"
]

def validate_columns(df):

    df.columns = [str(col).strip() for col in df.columns]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        return False, missing_cols

    return True, None

def copy_button(text, label="Copy"):
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText(`{text}`)"
        style="
            background-color:#1f6feb;
            color:white;
            border:none;
            padding:6px 12px;
            border-radius:6px;
            cursor:pointer;
        ">
        {label}
        </button>
        """,
        height=40,
    )
# --- 4. FUNGSI MEMBERSIHKAN DATA ---
def clean_val(x):
    if pd.isna(x) or str(x).strip() in ['-', 'ttu', 'TTU', '8888']:
        return 0.0
    if isinstance(x, str):
        x = x.replace(',', '.')
    try:
        return float(x)
    except:
        return 0.0

def process_data(df):
    df.columns = [str(col).strip() for col in df.columns]
    col_mapping = {
        "T '07.00": 'T07', "T '13.00": 'T13', "T '18.00": 'T18',
        'TRata2': 'T_Avg', 'TMax': 'T_Max', 'TMin': 'T_Min',
        'Curah Hujan (mm)': 'Rain', 'SS (%)': 'Sun',
        'Tekanan Udara (mb)': 'Pressure', 'RH07.00': 'RH07', 
        'RH13.00': 'RH13', 'RH18.00': 'RH18', 'RHRata2': 'RH_Avg',
        'Kec Rata2': 'WS_Avg', 'Arah Terbanyak': 'WD_Most',
        'Kec,Max': 'WS_Max', 'Arah': 'WD_Max'
    }
    df = df.rename(columns=col_mapping)
    if 'Tanggal' in df.columns:
        df['Tanggal_DT'] = pd.to_datetime(df['Tanggal'], dayfirst=True, errors='coerce')
    
    numeric_cols = ['T07', 'T13', 'T18', 'T_Avg', 'T_Max', 'T_Min', 'Rain', 'Sun', 
                    'Pressure', 'RH07', 'RH13', 'RH18', 'RH_Avg', 'WS_Avg', 
                    'WD_Most', 'WS_Max', 'WD_Max']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_val)
    return df

@st.cache_data
def load_station_file(station_name):

    file_path = f"data/{station_name}.csv"

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, sep=';')
            return process_data(df)
        except Exception as e:
            st.error(f"Gagal memproses file {station_name}: {e}")
            return None

    return None

# --- AI RISK ENGINE ---
def calculate_risk_score(df):
    score = 0

    if df['Rain'].sum() > 300:
        score += 30
    if df['T_Max'].max() >= 35:
        score += 20
    if df['WS_Max'].max() >= 25:
        score += 20
    if df['Pressure'].std() > 3:
        score += 15
    if df['RH_Avg'].mean() > 85:
        score += 15

    return min(score, 100)

def get_dominant_wind_direction(df):
    if 'WD_Most' not in df.columns or df['WD_Most'].empty:
        return "Tidak tersedia"

    dir_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    dir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    df = df.copy()
    df['WD_Label'] = pd.cut(
        df['WD_Most'],
        bins=dir_bins,
        labels=dir_labels,
        include_lowest=True
    )

    dominant = df['WD_Label'].value_counts().idxmax()

    return dominant

def push_to_github(file_content, file_name):

    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets["GITHUB_BRANCH"]

    api_url = f"https://api.github.com/repos/{repo}/contents/data/{file_name}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    # ✅ ENCODE DI SINI (WAJIB DI DALAM FUNGSI)
    encoded_content = base64.b64encode(file_content).decode("utf-8")

    # cek apakah file sudah ada
    response = requests.get(api_url, headers=headers)

    sha = None
    if response.status_code == 200:
        sha = response.json()["sha"]

    commit_message = f"Update data {file_name}"

    payload = {
        "message": commit_message,
        "content": encoded_content,
        "branch": branch
    }

    if sha:
        payload["sha"] = sha

    r = requests.put(api_url, headers=headers, json=payload)

    return r.status_code in [200, 201]


def internal_monitoring_report(df, station):

    if df.empty:
        return "Data tidak tersedia."

    risk_score = calculate_risk_score(df)

    # =============================
    # Tentukan Level Risiko
    # =============================
    if risk_score >= 70:
        level = "TINGGI"
        level_desc = "menunjukkan dinamika atmosfer yang signifikan dan memerlukan monitoring intensif."
    elif risk_score >= 40:
        level = "MENENGAH"
        level_desc = "menunjukkan adanya variabilitas cuaca yang perlu pemantauan berkala."
    else:
        level = "RENDAH"
        level_desc = "menunjukkan kondisi atmosfer relatif stabil."

    # =============================
    # Ambil periode otomatis
    # =============================
    start_date = df['Tanggal_DT'].min().strftime("%d %B %Y")
    end_date = df['Tanggal_DT'].max().strftime("%d %B %Y")

    # =============================
    # Identifikasi faktor dominan
    # =============================
    dominant_factors = []

    if df['Rain'].sum() > 300:
        dominant_factors.append("akumulasi curah hujan tinggi")

    if df['T_Max'].max() >= 35:
        dominant_factors.append("suhu maksimum ekstrem")

    if df['WS_Max'].max() >= 25:
        dominant_factors.append("kecepatan angin tinggi")

    if df['Pressure'].std() > 3:
        dominant_factors.append("fluktuasi tekanan signifikan")

    if df['RH_Avg'].mean() > 85:
        dominant_factors.append("kelembapan udara sangat tinggi")

    if dominant_factors:
        faktor_text = ", ".join(dominant_factors)
    else:
        faktor_text = "tidak terdapat parameter yang menunjukkan deviasi signifikan"

    # =============================
    # Statistik Ringkas
    # =============================
    suhu_avg = df['T_Avg'].mean()
    rain_total = df['Rain'].sum()
    wind_max = df['WS_Max'].max()
    pressure_std = df['Pressure'].std()
    dominant_wind = get_dominant_wind_direction(df)

    return f"""
LAPORAN MONITORING INTERNAL
UPT: {station}
Periode Analisis: {start_date} - {end_date}

INDIKATOR RISIKO CUACA
Risk Index: {risk_score}/100
Kategori Risiko: {level}

Analisis Risk Index:
Skor risiko yang diperoleh terutama dipengaruhi oleh {faktor_text}. 
Secara keseluruhan kondisi ini {level_desc}

Evaluasi Parameter:
• Rata-rata suhu: {suhu_avg:.1f}°C
• Total curah hujan: {rain_total:.1f} mm
• Angin maksimum: {wind_max:.1f} knot
• Variabilitas tekanan: {pressure_std:.2f} mb
• Arah angin dominan: {dominant_wind}
"""

# --- STATISTICAL ANALYSIS ENGINE ---
def statistical_analysis(df):
    results = {}

    numeric_cols = ['T_Avg', 'Rain', 'RH_Avg', 'Pressure', 'WS_Max']

    for col in numeric_cols:
        if col in df.columns and not df[col].empty:
            mean = df[col].mean()
            std = df[col].std()
            max_val = df[col].max()
            min_val = df[col].min()

            # Linear Trend (Slope)
            y = df[col].values
            x = np.arange(len(y))
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0]
            else:
                slope = 0

            # Anomaly detection (z-score)
            z_scores = (df[col] - mean) / std if std != 0 else 0
            anomaly_count = np.sum(np.abs(z_scores) > 2)

            results[col] = {
                "mean": mean,
                "std": std,
                "max": max_val,
                "min": min_val,
                "slope": slope,
                "anomaly_count": int(anomaly_count)
            }

    return results

def statistical_narrative(stats):
    text = "ANALISIS STATISTIK LANJUTAN:\n\n"

    for param, val in stats.items():

        trend_desc = "meningkat signifikan" if val["slope"] > 0.2 else \
                     "menurun signifikan" if val["slope"] < -0.2 else \
                     "relatif stabil"

        variability = "tinggi" if val["std"] > val["mean"]*0.2 else "normal"

        text += f"""
Parameter {param} memiliki rata-rata {val['mean']:.2f}
dengan variabilitas {variability}.
Tren parameter teridentifikasi {trend_desc}.
Jumlah kejadian anomali terdeteksi sebanyak {val['anomaly_count']} hari.

"""
    f""" Anomali artinya nilai menyimpang(lebih) dari nilai rata -rata """
    return text

# --- 5. SIDEBAR NAVIGATION ---
with st.sidebar:

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://www.meteoalor.id/assets/images/logo.png" width="110">
            <h3 style="margin-top:10px;">BMKG Jatim Monitor</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    selected_stn = st.selectbox("📍 Pilih Stasiun", STASIUN_LIST)
    df_raw = load_station_file(selected_stn)
    
    if df_raw is not None:
        years = sorted(df_raw['year'].unique().astype(int).tolist())
        sel_year = st.selectbox("📅 Tahun", years, index=len(years)-1)
        months = sorted(df_raw[df_raw['year'] == sel_year]['month'].unique().astype(int).tolist())
        sel_month = st.selectbox("📆 Bulan", ["Semua"] + months)
        if sel_month != "Semua":
            days = sorted(df_raw[(df_raw['year'] == sel_year) & (df_raw['month'] == sel_month)]['day'].unique().astype(int).tolist())
            sel_days = st.multiselect("🗓️ Pilih Hari", days)
        else:
            sel_days = []
        # ---------------------------------
        st.markdown("---")
        use_range = st.checkbox("📅 Gunakan Rentang Tanggal (Opsional)")

        if use_range:
            min_date = df_raw['Tanggal_DT'].min()
            max_date = df_raw['Tanggal_DT'].max()

            date_range = st.date_input(
        "Pilih Rentang Tanggal",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        else:
            date_range = None
        st.markdown("---")
        st.subheader("🔐 Admin Login")

        if "admin_logged" not in st.session_state:
            st.session_state.admin_logged = False

        admin_user = st.text_input("Username")
        admin_pass = st.text_input("Password", type="password")

        if st.button("Login"):
            if (admin_user == st.secrets["ADMIN_USER"] and
                admin_pass == st.secrets["ADMIN_PASS"]):
                st.session_state.admin_logged = True
                st.success("Login berhasil ✅")
            else:
                st.error("Username / Password salah ❌")

# ==========================
# ADMIN UPLOAD SECTION
# ==========================
        if st.session_state.admin_logged:

            st.markdown("---")
            st.subheader("📂 Upload Data Stasiun")

            target_station = st.selectbox(
                "Pilih Stasiun Tujuan",
                STASIUN_LIST
            )

            uploaded_file = st.file_uploader(
                "Upload File CSV",
                type=["csv"]
            )

            if uploaded_file is not None:

                try:
                    df_check = pd.read_csv(uploaded_file, sep=';')

            # =========================
            # PREVIEW DATA
            # =========================
                    st.markdown("### 👁️ Preview Data")
                    st.dataframe(df_check.head(20), use_container_width=True)

            # =========================
            # VALIDASI KOLOM
            # =========================
                    is_valid, missing = validate_columns(df_check)

                    if not is_valid:
                        st.error("❌ Format file tidak sesuai standar BMKG")
                        st.warning(f"Kolom yang hilang: {missing}")
                    else:
                        st.success("✅ Struktur kolom valid")
                        st.markdown("### 📊 Ringkasan Data")
                        st.write("Jumlah Baris:", len(df_check))
                        st.write("Rentang Tanggal:",
                             df_check["Tanggal"].min(),
                             "s/d",
                             df_check["Tanggal"].max())

                # Tombol push hanya muncul kalau valid
                        if st.button("🚀 Push ke GitHub"):

                            file_name = f"{target_station}.csv"
                            file_bytes = uploaded_file.getvalue()
                        
                            with st.spinner("Uploading ke GitHub..."):
                                success = push_to_github(file_bytes, file_name)
                    
                            if success:
                                st.success("✅ Upload & Push berhasil!")
                                st.cache_data.clear()
                                st.rerun()   
                            else:
                                st.error("❌ Gagal push ke GitHub")

                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")

# --- 6. MAIN CONTENT ---
if df_raw is not None:
    # FILTER LOGIC (Menambahkan .copy() untuk menghindari SettingWithCopyWarning)

    df_f = df_raw.copy()

    if use_range and date_range is not None and len(date_range) == 2:
    # PRIORITAS: RENTANG TANGGAL (override semua)
        start_date, end_date = date_range
        df_f = df_f[
            (df_f['Tanggal_DT'] >= pd.to_datetime(start_date)) &
            (df_f['Tanggal_DT'] <= pd.to_datetime(end_date))
        ].copy()

    else:
    # MODE LAMA (Tahun + Bulan + Hari)
        df_f = df_f[df_f['year'] == sel_year].copy()

        if sel_month != "Semua":
            df_f = df_f[df_f['month'] == sel_month].copy()

            if sel_days:
                df_f = df_f[df_f['day'].isin(sel_days)].copy()
        
    # --- RISK INDEX HEADER ---
    risk_score = calculate_risk_score(df_f)

    col_r1, col_r2 = st.columns([3,1])

    with col_r2:
        if risk_score >= 70:
            st.error(f"🔴 Risk Index {risk_score}/100")
        elif risk_score >= 40:
            st.warning(f"🟡 Risk Index {risk_score}/100")
        else:
            st.success(f"🟢 Risk Index {risk_score}/100")
    
    # KPI CARDS
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Suhu Avg", f"{df_f['T_Avg'].mean():.1f} °C")
    k2.metric("Suhu Max", f"{df_f['T_Max'].max():.1f} °C")
    k3.metric("RH Avg", f"{df_f['RH_Avg'].mean():.1f} %")
    k4.metric("Total Hujan", f"{df_f['Rain'].sum():.1f} mm")
    k5.metric("Angin Max", f"{df_f['WS_Max'].max():.1f} knt")

    # --- GRAFIK TREN SUHU ---
    st.subheader("🌡️ Tren Suhu")
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=df_f['Tanggal_DT'], y=df_f['T07'], name="07:00", line=dict(color="#00d4ff")))
    fig_t.add_trace(go.Scatter(x=df_f['Tanggal_DT'], y=df_f['T13'], name="13:00", line=dict(color="#ff4b4b")))
    fig_t.add_trace(go.Scatter(x=df_f['Tanggal_DT'], y=df_f['T18'], name="18:00", line=dict(color="#ffa500")))
    fig_t.update_layout(template="plotly_dark", hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_t, use_container_width=True)

    col_w, col_p = st.columns([1, 1])
    
    with col_w:
        st.subheader("🪁 Windrose (Arah Angin)")

    # ======================
    # 1️⃣ BIN ARAH (8 sektor)
    # ======================
        dir_bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
        dir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

        df_f.loc[:, 'WD_Label'] = pd.cut(
            df_f['WD_Most'],
            bins=dir_bins,
            labels=dir_labels,
            include_lowest=True
        )

    # ======================
    # 2️⃣ BIN KECEPATAN (warna)
    # ======================
        speed_bins = [0, 3, 6, 10, 15, 20, 50]
        speed_labels = ['0-3', '3-6', '6-10', '10-15', '15-20', '>20']

        df_f.loc[:, 'WS_Bin'] = pd.cut(
            df_f['WS_Avg'],   # 🔥 SUDAH SESUAI DATA KAMU
            bins=speed_bins,
            labels=speed_labels,
            include_lowest=True
        )

    # ======================
    # 3️⃣ GROUPING
    # ======================
        wind_data = (
            df_f.groupby(['WD_Label', 'WS_Bin'])
            .size()
           .reset_index(name='count')
        )

    # ======================
    # 4️⃣ PLOT
    # ======================
        fig_w = px.bar_polar(
            wind_data,
            r='count',
            theta='WD_Label',
            color='WS_Bin',
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )

        fig_w.update_layout(
            polar=dict(bgcolor="#161b22"),
            margin=dict(t=30, b=30, l=30, r=30),
            legend_title="Kecepatan (kt)"
        )

        st.plotly_chart(fig_w, use_container_width=True)

    with col_p:
        st.subheader("⏲️ Tekanan Udara (mb)")
        fig_pr = px.area(df_f, x='Tanggal_DT', y='Pressure', color_discrete_sequence=['#9b59b6'])
        fig_pr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pr, use_container_width=True)

    # --- ROW 3 ---
    c_rh, c_rn = st.columns(2)
    with c_rh:
        st.subheader("💧 Kelembapan Udara (RH %)")
        fig_rh = px.line(df_f, x='Tanggal_DT', y=['RH07', 'RH13', 'RH18', 'RH_Avg'], 
                         color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#ffffff'])
        fig_rh.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rh, use_container_width=True)
    with c_rn:
        st.subheader("🌧️ Curah Hujan (mm)")
        fig_rn = px.bar(df_f, x='Tanggal_DT', y='Rain', color='Rain', color_continuous_scale='Blues')
        fig_rn.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rn, use_container_width=True)

    # --- GAUGE RISK ---
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Risk Index"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "red" if risk_score>=70 else "orange" if risk_score>=40 else "green"}
        }
    ))
    f"""Risk Index dihitung berdasarkan kombinasi beberapa parameter meteorologis 
yang merepresentasikan tingkat dinamika atmosfer pada periode tersebut. 
Semakin tinggi skor, semakin besar potensi variabilitas atau kejadian cuaca signifikan.
"""
    st.plotly_chart(fig_gauge, use_container_width=True)

    month_label = sel_month
    report_text = internal_monitoring_report(df_f, selected_stn)

    st.text_area("Hasil Evaluasi Sistem", report_text, height=300)
    copy_button(report_text, "📋 Copy")

    
    # --- STATISTICAL INTELLIGENCE ---
    st.markdown("---")
    st.subheader("🧠 Statistical Intelligence Analysis")

    stats = statistical_analysis(df_f)
    stat_text = statistical_narrative(stats)

    st.text_area("Hasil Analisis Statistik", stat_text, height=350)
    copy_button(stat_text, "📋 Copy")
    with st.expander("👁️ Lihat Tabel Data"):
        st.dataframe(df_f, use_container_width=True)
        st.markdown("---")

else:
    st.warning("⚠️ Masukkan file excel ke folder 'data/' sesuai nama stasiun.")














