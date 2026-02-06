import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans

# 1. 페이지 설정
st.set_page_config(page_title="Incheon Airport Taxi Analysis", layout="wide")

st.title("🛫 인천공항(RKSI) 지상이동 시간 분석")
st.markdown("활주로 위치에 따른 주기장별 예상 이동 시간 및 군집화(Clustering) 시각화")

# 2. 데이터 로드 (캐싱을 사용하여 속도 향상)
@st.cache_data
def load_data():
    # CSV 파일이 없다면 예시 데이터를 생성하거나, 실제 파일을 업로드해야 함
    # 여기서는 업로드하신 파일과 유사한 형태의 가상 데이터를 로드한다고 가정
    # 실제 배포 시에는 'rksi_stands.csv'를 같은 폴더에 두고 pd.read_csv('rksi_stands.csv') 사용
    try:
        df = pd.read_csv('(2-3) AIRCRAFT PARKING DOCKING CHART_OCR.csv')
    except:
        st.error("데이터 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    return df

df = load_data()

# 3. 사이드바 설정 (사용자 입력)
st.sidebar.header("설정 (Configuration)")

# 활주로 선택
runways = {
    '33L (북풍/이륙)': (37.454167, 126.460833),
    '33R (북풍/이륙)': (37.456389, 126.464722),
    '34L (북풍/이륙)': (37.441111, 126.437778),
    '34R (북풍/이륙)': (37.443333, 126.441667),
    '15R (남풍/착륙)': (37.481667, 126.436389),
    '15L (남풍/착륙)': (37.483889, 126.440278),
}

selected_rwy = st.sidebar.selectbox("사용 활주로 선택", list(runways.keys()))
rwy_coord = runways[selected_rwy]
taxi_speed = st.sidebar.slider("평균 이동 속도 (Knots)", 10, 30, 15)

# 4. 분석 로직 (거리 및 시간 계산)
if not df.empty:
    # 거리 계산 함수
    def calculate_metrics(row):
        dy = abs(rwy_coord[0] - row['Lat']) * 111  # km
        dx = abs(rwy_coord[1] - row['Lon']) * 88   # km
        dist_km = dy + dx  # Manhattan Distance
        speed_kmh = taxi_speed * 1.852
        time_min = (dist_km / speed_kmh) * 60
        return time_min

    df['Est_Time'] = df.apply(calculate_metrics, axis=1)

    # 군집화 (K-Means)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['Est_Time']])
    
    # 군집 라벨링
    centroids = df.groupby('Cluster')['Est_Time'].mean().sort_values()
    labels = {centroids.index[0]: 'Short (단거리)', 
              centroids.index[1]: 'Medium (중거리)', 
              centroids.index[2]: 'Long (장거리)'}
    df['Cluster_Label'] = df['Cluster'].map(labels)

    # 5. 지도 시각화 (Folium)
    # 지도 중심을 인천공항으로 설정
    m = folium.Map(location=[37.46, 126.44], zoom_start=13)

    # 활주로 마커 표시 (빨간색 별)
    folium.Marker(
        location=rwy_coord,
        popup=f"Runway {selected_rwy}",
        icon=folium.Icon(color="red", icon="plane", prefix="fa")
    ).add_to(m)

    # 주기장 마커 표시 (군집별 색상)
    colors = {'Short (단거리)': 'green', 'Medium (중거리)': 'orange', 'Long (장거리)': 'red'}
    
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=5,
            popup=f"Stand: {row['Stand_ID']}\nTime: {row['Est_Time']:.1f}min",
            color=colors.get(row['Cluster_Label'], 'gray'),
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    col1, col2 = st.columns([3, 1])
    
    with col1:
        st_folium(m, width="100%", height=600)
    
    with col2:
        st.subheader("분석 결과")
        st.write(f"**선택 활주로:** {selected_rwy}")
        st.write(f"**평균 속도:** {taxi_speed} kts")
        
        # 군집별 통계
        stats = df.groupby('Cluster_Label')['Est_Time'].mean().reset_index()
        stats.columns = ['그룹', '평균소요시간(분)']
        st.dataframe(stats.sort_values('평균소요시간(분)'), hide_index=True)
        
        # Raw Data 다운로드
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("분석 데이터 다운로드", csv, "taxi_analysis.csv")

else:

    st.warning("데이터 파일을 로드할 수 없습니다. rksi_stands.csv 파일을 확인해주세요.")
