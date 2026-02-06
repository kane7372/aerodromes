import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Incheon Airport Zone Analysis", layout="wide")
st.title("🛫 인천공항(RKSI) 주기장 구역별 시각화")

# 1. 데이터 로드 (새로 만든 파일 rksi_stands_zoned.csv)
@st.cache_data
def load_data():
    try:
        return pd.read_csv('rksi_stands_zoned.csv')
    except:
        st.error("데이터 파일(rksi_stands_zoned.csv)이 없습니다. preprocess.py를 먼저 실행하세요.")
        return pd.DataFrame()

df = load_data()

# 2. 사이드바 설정
st.sidebar.header("설정 (Configuration)")

# 활주로 좌표 (상시 표시용)
runways = {
    '33L': (37.454167, 126.460833), '33R': (37.456389, 126.464722),
    '34L': (37.441111, 126.437778), '34R': (37.443333, 126.441667),
    '15R': (37.481667, 126.436389), '15L': (37.483889, 126.440278)
}

# 3. 구역(Category) 필터링
if not df.empty:
    # 데이터에 있는 카테고리 목록 가져오기
    all_categories = df['Category'].unique().tolist()
    
    st.sidebar.subheader("표시할 구역 선택")
    selected_zones = st.sidebar.multiselect(
        "구역(Zone) 필터",
        options=all_categories,
        default=all_categories # 기본적으로 모두 선택
    )
    
    # 선택된 구역만 필터링
    df_filtered = df[df['Category'].isin(selected_zones)]

    # 4. 지도 시각화
    m = folium.Map(location=[37.46, 126.44], zoom_start=13)

    # 활주로 표시 (회색 아이콘)
    for r_name, coord in runways.items():
        folium.Marker(
            location=coord,
            popup=f"RWY {r_name}",
            icon=folium.Icon(color='gray', icon='plane', prefix='fa')
        ).add_to(m)

    # 구역별 색상 매핑
    color_map = {
        'Passenger Apron': 'blue',       # 여객: 파랑
        'Cargo Apron': 'orange',         # 화물: 주황
        'Maintenance Apron': 'black',    # 정비: 검정/회색
        'Isolated Security Position': 'red', # 격리: 빨강 (경고색)
        'De-icing Apron': 'cyan'         # 제방빙: 하늘색
    }

    # 주기장 마커 찍기
    for _, row in df_filtered.iterrows():
        cat = row['Category']
        color = color_map.get(cat, 'green') # 지정 안 된 건 초록
        
        # 격리 주기장은 좀 더 눈에 띄게 표시
        radius = 8 if 'Isolated' in cat else 4
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"<b>[{cat}]</b><br>Stand: {row['Stand_ID']}",
            tooltip=f"{row['Stand_ID']} ({cat})"
        ).add_to(m)

    # 화면 구성
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st_folium(m, width="100%", height=600)
    
    with col2:
        st.subheader("범례 (Legend)")
        # 범례를 컬러 박스로 표시
        st.markdown(f"""
        - <span style='color:blue'>●</span> **Passenger Apron**: 여객 터미널
        - <span style='color:orange'>●</span> **Cargo Apron**: 화물 터미널
        - <span style='color:black'>●</span> **Maintenance**: 정비 주기장
        - <span style='color:red'>●</span> **Isolated**: 격리 주기장
        - <span style='color:cyan'>●</span> **De-icing**: 제방빙 패드
        """, unsafe_allow_html=True)
        
        st.divider()
        st.write(f"**총 표시 개수:** {len(df_filtered)}개")
        
        # 데이터 통계 표
        if not df_filtered.empty:
            stats = df_filtered['Category'].value_counts().reset_index()
            stats.columns = ['구역', '개수']
            st.dataframe(stats, hide_index=True)

else:
    st.warning("데이터가 없습니다.")
