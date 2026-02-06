import pandas as pd
import re

# 1. 파일 로드
file_path = '(2-3) AIRCRAFT PARKING DOCKING CHART_OCR.csv'

try:
    df = pd.read_csv(file_path)
    print(f"📂 파일 로드 성공: {file_path}")
except FileNotFoundError:
    print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
    exit()

def dms_to_decimal(dms_str):
    clean_str = re.sub(r"[^\d\.]", " ", str(dms_str))
    parts = clean_str.split()
    if len(parts) < 2: return None
    try:
        deg, min_val = float(parts[0]), float(parts[1])
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        val = deg + min_val/60 + sec/3600
        if (37 <= val <= 38) or (126 <= val <= 127): return val
        return None
    except: return None

extracted_data = []
rows = df.values.tolist()

# [핵심] 현재 읽고 있는 구역을 저장할 변수 (기본값: Passenger Apron)
current_zone = "Passenger Apron"

for r_idx, row in enumerate(rows):
    # 1. 행 전체 텍스트에서 구역(Zone) 키워드 감지
    row_text = " ".join([str(x) for x in row if pd.notna(x)])
    
    if "Cargo Apron" in row_text:
        current_zone = "Cargo Apron"
    elif "Maintenance Apron" in row_text:
        current_zone = "Maintenance Apron"
    elif "Isolated Security" in row_text:
        current_zone = "Isolated Security Position"
    elif "Deicing" in row_text or "De-icing" in row_text:
        current_zone = "De-icing Apron"
    elif "Apron" in row_text and "Cargo" not in row_text and "Maintenance" not in row_text:
        # Apron 1, Apron 2 등은 여객(Passenger)로 통일하거나 그대로 사용
        current_zone = "Passenger Apron"

    # 2. 좌표 추출 로직 (기존과 동일)
    lat_indices = [i for i, cell in enumerate(row) if isinstance(cell, str) and re.search(r"37[\D\d]*N", cell)]
    lon_indices = [i for i, cell in enumerate(row) if isinstance(cell, str) and re.search(r"126[\D\d]*E", cell)]
    
    for lat_idx in lat_indices:
        valid_lon = [i for i in lon_indices if i > lat_idx]
        if not valid_lon: continue
        lon_idx = valid_lon[0]
        
        lat_dec = dms_to_decimal(str(row[lat_idx]))
        lon_dec = dms_to_decimal(str(row[lon_idx]))
        
        if lat_dec and lon_dec:
            # Stand ID 추출
            stand_id = f"Spot_{len(extracted_data)+1}"
            match = re.search(r"^(\d+[A-Z]?)\s+37", str(row[lat_idx]))
            if match:
                stand_id = match.group(1)
            
            # 800번대는 De-icing으로 강제 분류 (보정)
            final_zone = current_zone
            if stand_id.startswith('8') and len(stand_id) >= 3:
                final_zone = "De-icing Apron"

            extracted_data.append({
                'Stand_ID': stand_id,
                'Lat': lat_dec,
                'Lon': lon_dec,
                'Category': final_zone  # 구역 정보 저장
            })

# 저장
if extracted_data:
    df_result = pd.DataFrame(extracted_data)
    df_result.to_csv('rksi_stands_zoned.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 분류 완료! 총 {len(df_result)}개 스팟 추출")
    print(df_result['Category'].value_counts()) # 구역별 개수 출력
else:
    print("❌ 데이터를 찾지 못했습니다.")
