import pandas as pd
import re
import os

# 1. 파일 찾기 (업로드한 파일명 자동 매칭)
target_file = '(2-3) AIRCRAFT PARKING DOCKING CHART_OCR.csv'
if not os.path.exists(target_file):
    print("❌ 오류: 원본 CSV 파일을 찾을 수 없습니다. 파일명을 확인해주세요.")
    exit()

# 헤더 없이 읽어서 모든 텍스트를 검색
df = pd.read_csv(target_file, header=None)
print(f"📂 파일 로드 성공: {target_file}")

# 2. 좌표 변환 함수 (OCR 노이즈 제거 및 유효성 검사)
def dms_to_decimal(dms_str):
    # 숫자와 점(.)을 제외한 노이즈 제거
    clean_str = re.sub(r"[^\d\.]", " ", str(dms_str))
    parts = clean_str.split()
    
    if len(parts) < 2: return None
    try:
        deg = float(parts[0])
        min_val = float(parts[1])
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        val = deg + min_val/60 + sec/3600
        
        # 인천공항 좌표 범위 (위도 37도 부근 OR 경도 126도 부근)
        # 주의: 위도와 경도를 하나의 함수로 처리하므로 'OR' 조건 사용
        if (37.0 <= val <= 38.0) or (126.0 <= val <= 127.0): 
            return val
        return None
    except: return None

# 3. 데이터 추출 및 분류
extracted_data = []
rows = df.values.tolist()
current_zone = "Passenger Apron" # 기본값

for r_idx, row in enumerate(rows):
    # 구역(Zone) 이름 감지
    row_text = " ".join([str(x) for x in row if pd.notna(x)])
    if "Cargo" in row_text and "Apron" in row_text: current_zone = "Cargo Apron"
    elif "Maintenance" in row_text: current_zone = "Maintenance Apron"
    elif "Isolated" in row_text: current_zone = "Isolated Security Position"
    elif "Deicing" in row_text or "De-icing" in row_text: current_zone = "De-icing Apron"
    elif "Apron" in row_text and "Cargo" not in row_text: current_zone = "Passenger Apron"

    # 위도(N)와 경도(E)가 있는 컬럼 찾기
    lat_indices = [i for i, c in enumerate(row) if isinstance(c, str) and re.search(r"37[\D\d]*N", c)]
    lon_indices = [i for i, c in enumerate(row) if isinstance(c, str) and re.search(r"126[\D\d]*E", c)]
    
    for lat_idx in lat_indices:
        # 짝이 되는 경도 컬럼 찾기
        valid_lon = [i for i in lon_indices if i > lat_idx]
        if not valid_lon: continue
        lon_idx = valid_lon[0]
        
        # [핵심] 한 셀에 들어있는 여러 줄(\n)을 분리
        lat_lines = str(row[lat_idx]).split('\n')
        lon_lines = str(row[lon_idx]).split('\n')
        
        # 줄 개수만큼 반복 처리
        count = min(len(lat_lines), len(lon_lines))
        for i in range(count):
            lat_txt = lat_lines[i]
            lon_txt = lon_lines[i]
            
            lat_dec = dms_to_decimal(lat_txt)
            lon_dec = dms_to_decimal(lon_txt)
            
            if lat_dec and lon_dec:
                # Stand ID 추출 (예: "101 37...N")
                stand_id = f"Spot"
                match = re.search(r"^(\d+[A-Z]?)\s+37", lat_txt.strip())
                if match:
                    stand_id = match.group(1)
                else:
                    # 바로 왼쪽 컬럼 등에서 번호 찾기 시도
                    if lat_idx > 0:
                        left_val = str(row[lat_idx-1]).split('\n')
                        if len(left_val) > i and re.match(r"^\d{1,3}[A-Z]?$", left_val[i].strip()):
                            stand_id = left_val[i].strip()

                # 800번대 스팟은 무조건 제방빙 패드로 분류
                final_zone = current_zone
                clean_id = re.sub(r"[^0-9]", "", stand_id)
                if clean_id.isdigit() and 800 <= int(clean_id) < 900:
                    final_zone = "De-icing Apron"

                extracted_data.append({
                    'Stand_ID': stand_id,
                    'Lat': lat_dec,
                    'Lon': lon_dec,
                    'Category': final_zone
                })

# 4. 저장
if extracted_data:
    df_result = pd.DataFrame(extracted_data)
    df_result = df_result.drop_duplicates(subset=['Lat', 'Lon']) # 중복 제거
    
    output_file = 'rksi_stands_zoned.csv'
    df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ 정제 완료! '{output_file}' 파일이 생성되었습니다.")
    print(f"   - 추출된 주기장 수: {len(df_result)}")
    print(f"   - 분류된 구역 현황:\n{df_result['Category'].value_counts()}")
else:
    print("❌ 데이터를 추출하지 못했습니다. 원본 파일을 확인해주세요.")
