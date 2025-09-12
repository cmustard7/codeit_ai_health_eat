"""
Kaggle 제출 파일 생성 스크립트
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_kaggle_submission(results_dir: str, output_path: str, format_type: str = "csv"):
    """
    Kaggle 제출 형식으로 변환
    
    Args:
        results_dir: 결과 JSON 파일들이 있는 디렉토리
        output_path: 출력 파일 경로
        format_type: 출력 형식 ("csv" 또는 "json")
    """
    # TODO: 실제 제출 파일 생성 로직 구현
    submission_data = []
    results_path = Path(results_dir)
    
    logger.info(f"Processing results from {results_dir}")
    
    # JSON 결과 파일들 처리
    for json_file in results_path.glob("*_result.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            image_id = json_file.stem.replace('_result', '')
            
            # 탐지된 알약들을 Kaggle 형식으로 변환
            for pill in result.get("detected_pills", []):
                submission_data.append({
                    'image_id': image_id,
                    'pill_id': pill["pill_id"],
                    'bbox': str(pill["bbox"]),  # [x1, y1, x2, y2] 형식
                    'class_name': pill["drug_name"],
                    'confidence': pill["confidence"]
                })
                
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    if not submission_data:
        logger.warning("No valid data found for submission")
        return
    
    # 출력 형식에 따라 저장
    if format_type.lower() == "csv":
        df = pd.DataFrame(submission_data)
        df.to_csv(output_path, index=False)
        logger.info(f"CSV submission file created: {output_path}")
        
    elif format_type.lower() == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON submission file created: {output_path}")
    
    logger.info(f"Submission file contains {len(submission_data)} detections")

def validate_submission_format(submission_file: str) -> bool:
    """
    제출 파일 형식 검증
    
    Args:
        submission_file: 제출 파일 경로
        
    Returns:
        is_valid: 유효성 여부
    """
    # TODO: 실제 형식 검증 로직 구현
    try:
        if submission_file.endswith('.csv'):
            df = pd.read_csv(submission_file)
            required_columns = ['image_id', 'pill_id', 'bbox', 'class_name', 'confidence']
            
            # 필수 컬럼 확인
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            
            # 데이터 타입 확인
            if df['confidence'].dtype not in ['float64', 'float32']:
                logger.error("Confidence column should be numeric")
                return False
            
            # 신뢰도 범위 확인
            if not df['confidence'].between(0, 1).all():
                logger.error("Confidence values should be between 0 and 1")
                return False
            
            logger.info(f"Submission file validation passed: {len(df)} rows")
            return True
            
        elif submission_file.endswith('.json'):
            with open(submission_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.error("JSON should contain a list of detections")
                return False
            
            # 첫 번째 항목 검증
            if data:
                required_keys = ['image_id', 'pill_id', 'bbox', 'class_name', 'confidence']
                missing_keys = [key for key in required_keys if key not in data[0]]
                if missing_keys:
                    logger.error(f"Missing keys in JSON: {missing_keys}")
                    return False
            
            logger.info(f"JSON submission file validation passed: {len(data)} detections")
            return True
            
    except Exception as e:
        logger.error(f"Error validating submission file: {e}")
        return False

def create_submission_summary(submission_file: str) -> dict:
    """
    제출 파일 요약 생성
    
    Args:
        submission_file: 제출 파일 경로
        
    Returns:
        summary: 제출 요약
    """
    # TODO: 실제 요약 생성 로직 구현
    try:
        if submission_file.endswith('.csv'):
            df = pd.read_csv(submission_file)
            
            summary = {
                'total_detections': len(df),
                'unique_images': df['image_id'].nunique(),
                'unique_classes': df['class_name'].nunique(),
                'avg_confidence': df['confidence'].mean(),
                'confidence_std': df['confidence'].std(),
                'pills_per_image': len(df) / df['image_id'].nunique() if df['image_id'].nunique() > 0 else 0,
                'class_distribution': df['class_name'].value_counts().to_dict()
            }
            
        elif submission_file.endswith('.json'):
            with open(submission_file, 'r') as f:
                data = json.load(f)
            
            confidences = [item['confidence'] for item in data]
            classes = [item['class_name'] for item in data]
            images = [item['image_id'] for item in data]
            
            summary = {
                'total_detections': len(data),
                'unique_images': len(set(images)),
                'unique_classes': len(set(classes)),
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'pills_per_image': len(data) / len(set(images)) if images else 0,
                'class_distribution': {cls: classes.count(cls) for cls in set(classes)}
            }
        
        logger.info(f"Submission summary created: {summary['total_detections']} detections")
        return summary
        
    except Exception as e:
        logger.error(f"Error creating submission summary: {e}")
        return {}

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Create Kaggle Submission File")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing result JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output submission file path")
    parser.add_argument("--format", type=str, choices=["csv", "json"], default="csv", help="Output format")
    parser.add_argument("--validate", action="store_true", help="Validate submission format")
    parser.add_argument("--summary", action="store_true", help="Create submission summary")
    
    args = parser.parse_args()
    
    # 제출 파일 생성
    create_kaggle_submission(args.results_dir, args.output, args.format)
    
    # 형식 검증
    if args.validate:
        is_valid = validate_submission_format(args.output)
        if is_valid:
            logger.info("✅ Submission file format is valid")
        else:
            logger.error("❌ Submission file format is invalid")
    
    # 요약 생성
    if args.summary:
        summary = create_submission_summary(args.output)
        if summary:
            summary_file = args.output.replace('.csv', '_summary.json').replace('.json', '_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info(f"Submission summary saved to {summary_file}")

if __name__ == "__main__":
    main()
