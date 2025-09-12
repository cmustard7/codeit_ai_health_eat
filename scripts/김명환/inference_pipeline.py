"""
메인 추론 파이프라인 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cv2
import torch
import yaml
from datetime import datetime
from pathlib import Path
import logging
import argparse

from src.models.yolo_detector import YOLODetector
from src.models.efficientnet_classifier import EfficientNetClassifier
from src.utils.validation import ModelValidator
from src.utils.metrics import SystemMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PillDetectionPipeline:
    """알약 탐지 통합 파이프라인"""
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.yolo_detector = YOLODetector(
            confidence_threshold=self.config['models']['yolo']['confidence_threshold']
        )
        self.efficientnet_classifier = EfficientNetClassifier(
            num_classes=self.config['models']['efficientnet']['num_classes']
        )
        self.validator = ModelValidator()
        self.system_metrics = SystemMetrics()
        
        logger.info("Pill detection pipeline initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        # TODO: 실제 설정 파일 로드 로직
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except:
            # 더미 설정 반환
            return {
                'models': {
                    'yolo': {'confidence_threshold': 0.5},
                    'efficientnet': {'num_classes': 50}
                }
            }
    
    def process_single_image(self, image_path: str) -> dict:
        """
        단일 이미지 처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            result: 처리 결과
        """
        # TODO: 실제 이미지 처리 로직 구현
        start_time = datetime.now()
        
        try:
            # 1. YOLO 객체 탐지
            logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            yolo_detections = self.yolo_detector.detect_pills(image)
            logger.info(f"YOLO detected {len(yolo_detections)} pills")
            
            # 2. EfficientNet 분류
            cropped_images = [det['cropped_image'] for det in yolo_detections]
            if cropped_images:
                classifications = self.efficientnet_classifier.batch_classify(cropped_images)
            else:
                classifications = []
            
            logger.info(f"Classification completed for {len(classifications)} pills")
            
            # 3. 결과 검증
            validated_results = self.validator.validate_detection_results(
                yolo_detections, classifications
            )
            
            # 4. 처리 시간 기록
            processing_time = (datetime.now() - start_time).total_seconds()
            self.system_metrics.record_inference_time(processing_time)
            
            # 5. 결과 포맷팅
            result = self._format_output(validated_results, image_path, processing_time)
            
            logger.info(f"Processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return self._create_error_result(image_path, str(e))
    
    def _format_output(self, validated_results: list, image_path: str, processing_time: float) -> dict:
        """결과 포맷팅"""
        # TODO: 실제 포맷팅 로직 구현
        output = {
            "image_info": {
                "image_path": image_path,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            },
            "analysis_summary": {
                "total_pills_detected": len(validated_results),
                "overall_confidence": self._calculate_overall_confidence(validated_results),
                "processing_status": "success"
            },
            "detected_pills": [],
            "system_performance": {
                "inference_time": processing_time,
                "memory_usage": "N/A"  # TODO: 실제 메모리 사용량 계산
            }
        }
        
        for result in validated_results:
            pill_info = {
                "pill_id": result['pill_id'],
                "drug_name": result['predicted_drug'],
                "confidence": result['final_confidence'],
                "bbox": result['bbox'],
                "classification_details": result.get('top3_predictions', [])
            }
            output["detected_pills"].append(pill_info)
        
        return output
    
    def _calculate_overall_confidence(self, results: list) -> float:
        """전체 신뢰도 계산"""
        if not results:
            return 0.0
        
        confidences = [result['final_confidence'] for result in results]
        return sum(confidences) / len(confidences)
    
    def _create_error_result(self, image_path: str, error_message: str) -> dict:
        """오류 결과 생성"""
        return {
            "image_info": {
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            },
            "analysis_summary": {
                "total_pills_detected": 0,
                "overall_confidence": 0.0,
                "processing_status": "error",
                "error_message": error_message
            },
            "detected_pills": [],
            "system_performance": {}
        }
    
    def process_batch(self, image_paths: list, output_dir: str = "outputs") -> list:
        """
        배치 처리
        
        Args:
            image_paths: 이미지 경로 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            batch_results: 배치 처리 결과
        """
        # TODO: 실제 배치 처리 로직 구현
        Path(output_dir).mkdir(exist_ok=True)
        batch_results = []
        
        logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            result = self.process_single_image(image_path)
            batch_results.append(result)
            
            # 개별 결과 저장
            output_file = Path(output_dir) / f"{Path(image_path).stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 배치 요약 저장
        summary = self._create_batch_summary(batch_results)
        summary_file = Path(output_dir) / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch processing completed. Results saved to {output_dir}")
        return batch_results
    
    def _create_batch_summary(self, batch_results: list) -> dict:
        """배치 요약 생성"""
        # TODO: 실제 배치 요약 로직 구현
        total_images = len(batch_results)
        successful_images = sum(1 for r in batch_results if r['analysis_summary']['processing_status'] == 'success')
        total_pills = sum(r['analysis_summary']['total_pills_detected'] for r in batch_results)
        
        performance_stats = self.system_metrics.calculate_system_performance()
        
        summary = {
            "batch_info": {
                "total_images": total_images,
                "successful_images": successful_images,
                "failed_images": total_images - successful_images,
                "total_pills_detected": total_pills,
                "timestamp": datetime.now().isoformat()
            },
            "performance_summary": performance_stats,
            "success_rate": successful_images / total_images if total_images > 0 else 0.0
        }
        
        return summary

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Pill Detection Pipeline")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--batch", type=str, help="Directory containing images")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = PillDetectionPipeline(args.config)
    
    if args.image:
        # 단일 이미지 처리
        result = pipeline.process_single_image(args.image)
        
        # 결과 저장
        output_file = Path(args.output) / f"{Path(args.image).stem}_result.json"
        Path(args.output).mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Result saved to {output_file}")
        
    elif args.batch:
        # 배치 처리
        image_dir = Path(args.batch)
        image_paths = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(image_dir.glob(ext))
        
        if not image_paths:
            logger.error(f"No images found in {args.batch}")
            return
        
        batch_results = pipeline.process_batch(
            [str(p) for p in image_paths], 
            args.output
        )
        
        logger.info(f"Batch processing completed: {len(batch_results)} images processed")
        
    else:
        logger.error("Please specify either --image or --batch option")

if __name__ == "__main__":
    main()
