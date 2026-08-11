import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pythainlp.tokenize import word_tokenize

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Fraud-Scout | Retail Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. NLP Engine Setup (Cached for performance)
# ==========================================
def thai_tokenizer(text):
    return word_tokenize(text, engine='newmm', keep_whitespace=False)

@st.cache_resource
def load_nlp_engine():
    # อัปเดตคลังข้อมูลให้ครอบคลุมภาษาพูดและสถานการณ์จริงของธุรกิจค้าปลีก
    cases_dict = {
        "Case 1": (
            "ขายของตอนดึก แอบขายตอนร้านปิด มีบิลโผล่มาตอนเที่ยงคืน ยอดเข้าตอนกลางคืน นอกเวลาทำงาน แอบเปิดเครื่องคิดเงิน "
            "ดึก กลางคืน ปิดร้าน ตีหนึ่ง ตีสอง ตีสาม หลังเวลา เลิกงาน นอกเวลา ยามวิกาล บิลแปลกๆ ตอนมืด มียอดเข้าตอนร้านปิด "
            "midnight sales out of hours closed shop late night after hours odd time"
        ),
        "Case 2": (
            "รายการหาย ยอดรวมหาย ยอดหาย หายไป ข้อมูลหาย ยกเลิกรายการ ลบของออกจากบิล แอบลบรายการแล้วรับเงินสด "
            "ลบบิลทิ้ง เงินในเก๊ะไม่ครบ ยกเลิกออเดอร์ แอบขโมยเงิน ข้อมูลไม่ครบ ข้อมูลว่าง ไม่มีชื่อสินค้า ไม่มีราคา ไม่มีจำนวน "
            "void โว้ยบิล แอบลบ ลบรายการ บิลแหว่ง แอบดึงเงิน แอบหยิบเงิน ช่องโหว่ "
            "ghost transactions missing data voided bills null values deleted empty no item cash stolen"
        ),
        "Case 3": (
            "ยอดเงินไม่ตรงกับของที่ขาย แอบลดราคาให้เพื่อน ขายของถูกกว่าป้าย คำนวณเงินผิด เงินได้น้อยกว่าของที่ออกไป "
            "ยอดรวมไม่ตรง ราคาไม่ตรง แก้ราคา เปลี่ยนราคา แอบลด ลดราคา ยอดเพี้ยน คิดเงินผิด โกงราคา เงินขาด ยอดคูณไม่ตรง "
            "price manipulation mismatch discount abuse changed unit price calculation error wrong total reduced"
        )
    }
    case_names = list(cases_dict.keys())
    case_descriptions = list(cases_dict.values())

    vectorizer = TfidfVectorizer(tokenizer=thai_tokenizer)
    tfidf_matrix = vectorizer.fit_transform(case_descriptions)

    return vectorizer, tfidf_matrix, case_names

vectorizer, tfidf_matrix, case_names = load_nlp_engine()

def match_fraud_case(user_query):
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < 0.05:
        return None, best_score
    return case_names[best_idx], best_score

# ==========================================
# 3. Translation Dictionary
# ==========================================
texts = {
    "TH": {
        "hero_subtitle": "ระบบปัญญาประดิษฐ์ตรวจจับความผิดปกติของยอดขายและป้องกันการทุจริตสำหรับธุรกิจค้าปลีก",
        "trust_badge": "**Zero-Data Retention (นโยบายความปลอดภัยสูงสุด):** ข้อมูล CSV ที่ท่านอัปโหลดจะถูกประมวลผลบนหน่วยความจำชั่วคราวเพื่อการวิเคราะห์ในเซสชันนี้เท่านั้น และจะ **ถูกลบทิ้งทันที**",
        "cases_header": "📚 รูปแบบการทุจริตที่พบบ่อยในธุรกิจค้าปลีก",
        "case1_title": "🚨 Case 1: ยอดขายผีหลอก (Midnight Sales)",
        "case1_desc": "มีการทำรายการยอดขายในช่วงเวลาที่ร้านปิดไปแล้ว หรือนอกเวลาทำการปกติ ซึ่งอาจเกิดจากการลักลอบเปิดเครื่อง POS",
        "case2_title": "🚨 Case 2: บิลสูญหาย (Ghost Transactions)",
        "case2_desc": "ข้อมูลสำคัญเช่น ราคาสินค้า จำนวน หายไปจากระบบ (Null Values) บ่งชี้ถึงการ Void บิลที่ผิดปกติเพื่อรับเงินสด",
        "case3_title": "🚨 Case 3: แก้ไขราคา (Price Manipulation)",
        "case3_desc": "ยอดเงินสุทธิ (Total Spent) ไม่ตรงกับราคาสินค้าคูณด้วยจำนวน อาจเกิดจากการแอบปรับลดราคาให้คนรู้จัก",
        "tool_header": "🔍 เครื่องมือตรวจสอบความเสี่ยง (Diagnostic Tool)",
        "tool_desc": "อธิบายพฤติกรรมต้องสงสัยที่คุณกังวล และอัปโหลดไฟล์เพื่อเริ่มต้นการตรวจสอบ",
        "search_header": "1. อธิบายปัญหาที่พบ",
        "search_placeholder": "เช่น พนักงานชอบ void บิลตอนดึก หรือ ยอดเงินไม่ตรง...",
        "search_btn": "วิเคราะห์ปัญหา",
        "upload_header": "2. อัปโหลด Transaction Logs",
    },
    "EN": {
        "hero_subtitle": "AI-Powered Retail Anomaly Detection and Fraud Prevention System",
        "trust_badge": "**Zero-Data Retention (Maximum Security Policy):** Your uploaded CSV data is processed purely in-memory for this session and is **deleted immediately** after.",
        "cases_header": "📚 Common Retail Fraud Patterns",
        "case1_title": "🚨 Case 1: Midnight Sales",
        "case1_desc": "Transactions occurring outside of normal business hours. This could indicate unauthorized POS access.",
        "case2_title": "🚨 Case 2: Ghost Transactions",
        "case2_desc": "Missing critical data (Null Values) like price or quantity, indicating suspicious voided bills for cash.",
        "case3_title": "🚨 Case 3: Price Manipulation",
        "case3_desc": "Total spent does not match unit price multiplied by quantity. Indicates unauthorized discounts.",
        "tool_header": "🔍 Diagnostic Tool",
        "tool_desc": "Describe the suspicious behavior and upload your data to begin the risk assessment.",
        "search_header": "1. Describe the Suspicious Behavior",
        "search_placeholder": "e.g., Cashier voiding bills late at night...",
        "search_btn": "Analyze Problem",
        "upload_header": "2. Upload Transaction Logs",
    }
}

# ==========================================
# 4. Sidebar & Settings
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    lang = st.radio("🌐 Language / ภาษา", ["TH", "EN"], horizontal=True)
    t = texts[lang]
    st.divider()
    st.info("Developed by: Titipon Tawong")

# ==========================================
# 5. Hero Section (Custom Center Layout)
# ==========================================
st.markdown(f"""
    <div style='text-align: center; padding-top: 1rem; padding-bottom: 2rem;'>
        <h1 style='color: #FF4B4B; font-size: 4rem; margin-bottom: 0px;'>
            🛡️ Fraud-Scout
        </h1>
        <p style='font-size: 1.2rem; color: #888888;'>
            {t['hero_subtitle']}
        </p>
    </div>
""", unsafe_allow_html=True)

st.success(t["trust_badge"], icon="🔒")
st.divider()

# ==========================================
# 6. Display Cases (Educational Section)
# ==========================================
st.markdown(f"<h3 style='text-align: center;'>{t['cases_header']}</h3><br>", unsafe_allow_html=True)

case_col1, case_col2, case_col3 = st.columns(3)

with case_col1:
    st.info(f"**{t['case1_title']}**\n\n{t['case1_desc']}")

with case_col2:
    st.error(f"**{t['case2_title']}**\n\n{t['case2_desc']}")

with case_col3:
    st.warning(f"**{t['case3_title']}**\n\n{t['case3_desc']}")

st.divider()

# ==========================================
# 7. Diagnostic Tool (NLP + Upload)
# ==========================================
st.markdown(f"### {t['tool_header']}")
st.write(t["tool_desc"])
st.write("")

# Initialize session state for matched case
if 'matched_case' not in st.session_state:
    st.session_state.matched_case = None

tool_col1, tool_col2 = st.columns([1.2, 1])

with tool_col1:
    st.subheader(t["search_header"])
    user_query = st.text_area("Query", label_visibility="collapsed", placeholder=t["search_placeholder"], height=100)

    if st.button(t["search_btn"], type="primary", use_container_width=True):
        if user_query:
            matched_case, score = match_fraud_case(user_query)
            if matched_case:
                st.session_state.matched_case = matched_case
                st.success(f"✅ **Matched:** {matched_case}")
                st.caption(f"Confidence Score: {score:.2f}")
                msg = f"ระบบวิเคราะห์ว่าปัญหาของคุณตรงกับความเสี่ยงใน **{matched_case}** กรุณาอัปโหลดไฟล์ด้านขวาเพื่อเริ่มตรวจสอบเงื่อนไขนี้" if lang == "TH" else f"Your issue matches **{matched_case}**. Please upload your file on the right."
                st.write(msg)
            else:
                st.session_state.matched_case = None
                st.warning(
                    "ไม่สามารถระบุ Case ที่ชัดเจนได้ กรุณาลองอธิบายปัญหาด้วยคีย์เวิร์ดที่เจาะจงขึ้นครับ" if lang == "TH" else "Could not determine a specific case. Please try again with more specific keywords.")
        else:
            st.warning("Please enter a description first." if lang == "EN" else "กรุณาพิมพ์อธิบายปัญหาก่อนครับ")

with tool_col2:
    st.subheader(t["upload_header"])
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")

# ==========================================
# 8. Data Processing Logic
# ==========================================
if uploaded_file is not None and st.session_state.matched_case is not None:
    st.divider()
    st.markdown("### 📊 ผลการตรวจสอบ (Analysis Results)")

    try:
        df = pd.read_csv(uploaded_file)
        target_case = st.session_state.matched_case
        anomalies = pd.DataFrame()

        if target_case == "Case 1":
            st.info(
                "ตรวจสอบยอดขายนอกเวลาทำการ (เที่ยงคืน - 6 โมงเช้า)..." if lang == "TH" else "Checking for after-hours transactions...")
            if 'Transaction Time' in df.columns:
                df['Time_Obj'] = pd.to_datetime(df['Transaction Time'], format='%H:%M:%S', errors='coerce').dt.time
                mask = (df['Time_Obj'] >= pd.to_datetime('00:00:00', format='%H:%M:%S').time()) & (
                            df['Time_Obj'] <= pd.to_datetime('06:00:00', format='%H:%M:%S').time())
                anomalies = df[mask].drop(columns=['Time_Obj'])
            else:
                st.error(
                    "ไม่พบคอลัมน์ 'Transaction Time' ในข้อมูล" if lang == "TH" else "Column 'Transaction Time' not found.")

        elif target_case == "Case 2":
            st.error(
                "ตรวจสอบรายการที่ข้อมูลสูญหาย (Ghost Transactions)..." if lang == "TH" else "Checking for missing data (Ghost Transactions)...")
            mask = df['Price Per Unit'].isna() | df['Quantity'].isna() | df['Total Spent'].isna()
            anomalies = df[mask]

        elif target_case == "Case 3":
            st.warning(
                "ตรวจสอบความผิดปกติของยอดเงิน (Price Manipulation)..." if lang == "TH" else "Checking for price manipulation...")
            mask_not_null = df[['Price Per Unit', 'Quantity', 'Total Spent']].notnull().all(axis=1)
            # Use np.isclose to handle floating point precision issues safely
            mask_manipulated = ~np.isclose(df['Price Per Unit'] * df['Quantity'], df['Total Spent'], atol=0.01)
            anomalies = df[mask_not_null & mask_manipulated]

        if not anomalies.empty:
            msg = f"🚨 พบรายการผิดปกติจำนวน {len(anomalies)} รายการ" if lang == "TH" else f"🚨 Found {len(anomalies)} anomalous transactions."
            st.error(msg)
            st.dataframe(anomalies, use_container_width=True)
        else:
            msg = "✅ ไม่พบรายการผิดปกติในเงื่อนไขนี้" if lang == "TH" else "✅ No anomalies found for this case."
            st.success(msg)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}" if lang == "TH" else f"Error reading file: {e}")