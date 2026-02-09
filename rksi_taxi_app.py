import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import os

st.set_page_config(page_title="Incheon Airport Zone Map", layout="wide")

# 1. 데이터 로드
@st.cache_data
def load_data():
    file_path = 'rksi_stands_zoned.csv'
    if not os.path.exists(file_path):
        st.error(f"🚨 '{file_path}' 파일이 없습니다! 먼저 데이터 복구 코드를 실행해주세요.")
        return pd.DataFrame()
    return pd.read_csv(file_path)

df = load_data()

st.title("🛫 인천공항(RKSI) 주기장 8개 구역별 상세 지도")

if df.empty:
    st.stop()

# 2. 사이드바 설정 (runways 부분만 교체하세요)
st.sidebar.header("설정 (Configuration)")

# 활주로 좌표 (인천공항 4활주로 시스템 반영)
# L/R은 방향(Left/Right), 숫자는 방위각입니다.
runways = {
    # [제3, 4 활주로 / 서쪽]
    '33L': (37.4541, 126.4608), '15R': (37.4816, 126.4363), # 서로 반대편
    '33R': (37.4563, 126.4647), '15L': (37.4838, 126.4402), # 서로 반대편
    
    # [제1, 2 활주로 / 동쪽]
    '34L': (37.4411, 126.4377), '16R': (37.4680, 126.4130), # 서로 반대편
    '34R': (37.4433, 126.4416), '16L': (37.4700, 126.4170)  # 서로 반대편
}
# 3. 구역(Category) 필터링
# 실제 데이터에 존재하는 구역만 정렬해서 표시
all_categories = sorted(df['Category'].unique().tolist())

# 우선순위 정렬 (Apron -> Cargo -> Others)
sort_order = [
    'Apron 1', 'Apron 2', 'Apron 3', 
    'Cargo Apron 1', 'Cargo Apron 2', 
    'Maintenance Apron', 'De-icing Apron', 'Isolated Security Position'
]
all_categories = sorted(all_categories, key=lambda x: sort_order.index(x) if x in sort_order else 99)

st.sidebar.subheader("표시할 구역 선택")
selected_zones = st.sidebar.multiselect(
    "구역(Zone) 필터",
    options=all_categories,
    default=all_categories
)

df_filtered = df[df['Category'].isin(selected_zones)]

# 4. 지도 시각화
# 중심 좌표를 데이터의 평균 위치로 자동 조정
center_lat = df['Lat'].mean() if not df.empty else 37.46
center_lon = df['Lon'].mean() if not df.empty else 126.44
m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

# 활주로 표시
for r_name, coord in runways.items():
    folium.Marker(
        location=coord,
        popup=f"RWY {r_name}",
        icon=folium.Icon(color='gray', icon='plane', prefix='fa')
    ).add_to(m)

# 8개 구역별 색상 매핑 (Color Mapping)
color_map = {
    'Apron 1': 'blue',          # T1: 파랑
    'Apron 2': 'green',         # 탑승동: 초록
    'Apron 3': 'purple',        # T2: 보라
    'Cargo Apron 1': 'orange',  # 화물1: 주황
    'Cargo Apron 2': 'darkred', # 화물2: 진한 빨강
    'Maintenance Apron': 'black', # 정비: 검정
    'De-icing Apron': 'aqua', # 제방빙: 아쿠아
    'Isolated Security Position': 'red' # 격리: 빨강 (경고)
}

# 주기장 마커 찍기
for _, row in df_filtered.iterrows():
    cat = row['Category']
    # 매핑된 색상이 없으면 회색(gray) 사용
    color = color_map.get(cat, 'gray') 
    
    # 제방빙장은 더 크고 눈에 띄게
    radius = 10 if 'De-icing Apron' in cat else 4
    
    folium.CircleMarker(
        location=[row['Lat'], row['Lon']],
        radius=radius,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"<b>[{cat}]</b><br>Stand: {row['Stand_ID']}",
        tooltip=f"{row['Stand_ID']}"
    ).add_to(m)

# 화면 구성
col1, col2 = st.columns([3, 1])

with col1:
    st_folium(m, width="100%", height=700)

with col2:
    st.subheader("범례 (Legend)")
    
    # 범례 HTML 생성
    legend_html = ""
    for zone in all_categories:
        c = color_map.get(zone, 'gray')
        legend_html += f"- <span style='color:{c}'>●</span> **{zone}**<br>"
    
    st.markdown(legend_html, unsafe_allow_html=True)
    
    st.divider()
    st.write(f"**총 표시 개수:** {len(df_filtered)}개")
    
    if not df_filtered.empty:
        stats = df_filtered['Category'].value_counts().reindex(all_categories).fillna(0).astype(int).reset_index()
        stats.columns = ['구역', '개수']
        st.dataframe(stats, hide_index=True)




