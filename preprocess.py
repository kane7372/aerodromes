import pandas as pd
import re

# 1. 원본 파일 읽기 (사용자님이 업로드하신 파일명과 일치해야 함)
# 만약 파일명이 다르다면 아래 이름을 실제 파일명으로 수정해주세요.
input_file = '(2-3) AIRCRAFT PARKING DOCKING CHART_OCR.xlsx - Table 1.csv'

try:
    df = pd.read_csv(input_file)
    print(f"파일 로드 성공: {input_file}")
except FileNotFoundError:
    print(f"오류: '{input_file}' 파일을 찾을 수 없습니다. 파일명을 확인해주세요.")
    exit()

# 2. 좌표 변환 함수 (OCR 텍스트 -> 숫자)
def dms_to_decimal(dms_str):
    clean_str = re.sub(r"[^\d\.]", " ", str(dms_str))
    parts = clean_str.split()
    if len(parts) < 2: return None
    try:
        deg, min_val = float(parts[0]), float(parts[1])
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        # 인천공항 위도(37), 경도(126) 범위 체크
        decimal_val = deg + min_val/60 + sec/3600
        if 37 <= decimal_val <= 38 or 126 <= decimal_val <= 127:
            return decimal_val
        return None
    except: return None

# 3. 데이터 추출
extracted_data = []
rows = df.values.tolist()

for r_idx, row in enumerate(rows):
    # '37...N' 패턴(위도)과 '126...E' 패턴(경도) 찾기
    lat_indices = [i for i, cell in enumerate(row) if isinstance(cell, str) and re.search(r"37[\D\d]*N", cell)]
    lon_indices = [i for i, cell in enumerate(row) if isinstance(cell, str) and re.search(r"126[\D\d]*E", cell)]
    
    for lat_idx in lat_indices:
        # 위도 셀 바로 오른쪽에 있는 경도 셀 매칭
        valid_lon = [i for i in lon_indices if i > lat_idx]
        if not valid_lon: continue
        lon_idx = valid_lon[0]
        
        # 한 셀에 엔터로 여러 줄이 있는 경우 처리
        lat_lines = str(row[lat_idx]).split('\n')
        lon_lines = str(row[lon_idx]).split('\n')
        
        for i in range(min(len(lat_lines), len(lon_lines))):
            lat_dec = dms_to_decimal(lat_lines[i])
            lon_dec = dms_to_decimal(lon_lines[i])
            
            if lat_dec and lon_dec and (37.4 < lat_dec < 37.6) and (126.3 < lon_dec < 126.6):
                # 주기장 번호(Stand ID) 추출 시도
                stand_match = re.search(r"^(\d+[A-Z]?)\s+37", lat_lines[i])
                stand_id = stand_match.group(1) if stand_match else f"Spot_{len(extracted_data)+1}"
                
                extracted_data.append({
                    'Stand_ID': stand_id,
                    'Lat': lat_dec,
                    'Lon': lon_dec
                })

# 4. 결과 저장
if extracted_data:
    df_result = pd.DataFrame(extracted_data)
    output_filename = 'rksi_stands.csv'
    df_result.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"변환 완료! '{output_filename}' 파일이 생성되었습니다. (총 {len(df_result)}개 데이터)")
    print("이제 Streamlit 앱을 실행하시면 됩니다.")
else:
    print("데이터 추출 실패. 원본 파일의 형식을 확인해주세요.")
