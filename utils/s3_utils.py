import boto3
from dotenv import load_dotenv
import os


class S3Utils:
    BUCKET_NAME = "big9-project-01"  # 고정된 버킷 이름

    def __init__(self, region_name=None):
        """
        S3 유틸리티 클래스 초기화

        Args:
            region_name (str): AWS 리전 이름 (기본값: .env 파일에서 로드)
        """
        # .env 파일에서 환경 변수 로드
        load_dotenv()

        # AWS 자격증명 로드
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS 자격증명이 .env 파일에 설정되지 않았습니다.")

        # S3 클라이언트 초기화
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def list_files(self):
        """
        고정된 S3 버킷의 파일 목록 가져오기

        Returns:
            list: 버킷 내 파일 키(이름) 목록
        """
        try:
            # ListObjectsV2 API를 사용하여 객체 목록 가져오기
            response = self.s3_client.list_objects_v2(Bucket=self.BUCKET_NAME)

            # 버킷에 객체가 없는 경우 빈 리스트 반환
            if "Contents" not in response:
                return []

            # 파일 키(이름) 추출
            return [obj["Key"] for obj in response["Contents"]]

        except self.s3_client.exceptions.NoSuchBucket:
            # 버킷이 존재하지 않는 경우 빈 리스트 반환
            print(f"버킷 '{self.BUCKET_NAME}'이(가) 존재하지 않습니다.")
            return []
        except Exception as e:
            # 기타 예외 처리
            print(f"파일 목록을 가져오는 중 오류 발생: {e}")
            return []

    def upload_file(self, key: str, content: str):
        """
        S3 버킷에 파일 업로드

        Args:
            key (str): S3에 저장될 파일의 키(경로)
            content (str): 업로드할 파일 내용
        """
        try:
            # 문자열 내용을 바이트로 인코딩하여 업로드
            self.s3_client.put_object(
                Bucket=self.BUCKET_NAME, Key=key, Body=content.encode("utf-8")
            )
        except Exception as e:
            print(f"파일 업로드 중 오류 발생: {e}")
            raise

    def download_file(self, key: str, local_path: str) -> None:
        """
        S3 버킷에서 파일 다운로드

        Args:
            key (str): 다운로드할 파일의 키(경로)
            local_path (str): 파일을 저장할 로컬 경로

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
        """
        print(f"Downloading file with key: {key}, local_path: {local_path}")
        try:
            # 파일 객체 가져오기
            response = self.s3_client.get_object(Bucket=self.BUCKET_NAME, Key=key)
            content = response["Body"].read()

            # 디렉토리 경로 생성
            directory = os.path.dirname(local_path)
            if directory:  # 경로가 비어 있지 않을 때만 디렉토리 생성
                os.makedirs(directory, exist_ok=True)

            # 파일 저장
            with open(local_path, "wb") as f:
                f.write(content)

            print(f"파일이 {local_path}에 성공적으로 다운로드되었습니다.")

        except self.s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"파일 '{key}'을(를) 찾을 수 없습니다.")
        except Exception as e:
            print(f"파일 다운로드 중 오류 발생: {e}")
            raise
