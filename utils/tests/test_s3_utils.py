import unittest
from moto import mock_aws
import boto3
from utils.s3_utils import S3Utils
import os
import tempfile
import shutil


class TestS3Utils(unittest.TestCase):
    def setUp(self):
        """
        S3 유틸리티 테스트를 위한 모의 AWS 환경 설정.

        이 메서드는 모의 S3 환경을 초기화하고, 테스트 버킷을 생성하며,
        테스트용 파일을 업로드합니다.
        """
        # 모의 AWS 환경 시작
        self.mock = mock_aws()
        self.mock.start()

        # 모의 S3 클라이언트 생성
        self.s3_client = boto3.client("s3", region_name="ap-northeast-2")

        # 사전 정의된 버킷 이름 사용
        self.bucket_name = S3Utils.BUCKET_NAME

        # 테스트 버킷 생성
        self.s3_client.create_bucket(
            Bucket=self.bucket_name,
            CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"},
        )

        # 테스트 파일 업로드
        self.test_files = [
            {"key": "test_file_1.txt", "content": "Test content 1"},
            {"key": "test_file_2.txt", "content": "Test content 2"},
            {"key": "nested/test_file_3.txt", "content": "Test content 3"},
        ]

        for file in self.test_files:
            self.s3_client.put_object(
                Bucket=self.bucket_name, Key=file["key"], Body=file["content"]
            )

        # S3Utils 초기화
        self.s3_utils = S3Utils()

        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()

    def test_list_files(self):
        """
        list_files가 버킷 내 모든 파일을 올바르게 검색하는지 테스트.

        메서드가 중첩 디렉터리의 파일을 포함한 전체 파일 키 목록을
        반환하는지 검증합니다.
        """
        files = self.s3_utils.list_files()

        # 비교를 위해 파일 키만 추출
        expected_files = [file["key"] for file in self.test_files]

        # 반환된 목록에 모든 예상 파일이 있는지 확인
        self.assertCountEqual(files, expected_files)

    def test_file_upload(self):
        """
        S3 버킷에 새 파일 업로드 테스트.

        파일을 성공적으로 업로드하고 파일 목록에 추가되는지 검증합니다.
        """
        new_file_key = "new_test_file.txt"
        new_file_content = "새로운 테스트 콘텐츠"

        # 새 파일 업로드
        self.s3_utils.upload_file(new_file_key, new_file_content)

        # 파일이 파일 목록에 있는지 확인
        files = self.s3_utils.list_files()
        self.assertIn(new_file_key, files)

    def test_file_download(self):
        """
        S3 버킷에서 파일 다운로드 테스트.

        1. 파일 내용만 문자열로 받기
        2. 파일을 로컬에 저장하기
        """
        test_file = self.test_files[0]

        # 1. 파일 내용만 문자열로 받기 테스트
        downloaded_content = self.s3_utils.download_file(test_file["key"])
        self.assertEqual(downloaded_content, test_file["content"])

        # 2. 파일을 로컬에 저장하기 테스트
        local_path = os.path.join(self.temp_dir, "downloaded_file.txt")
        downloaded_content = self.s3_utils.download_file(test_file["key"], local_path)

        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(local_path))

        # 저장된 파일 내용 확인
        with open(local_path, "r") as f:
            saved_content = f.read()
        self.assertEqual(saved_content, test_file["content"])

    def test_non_existent_file(self):
        """
        존재하지 않는 파일 작업 처리 테스트.

        존재하지 않는 파일을 다운로드하거나 삭제하려 할 때
        적절한 예외가 발생하는지 확인합니다.
        """
        non_existent_key = "non_existent_file.txt"

        with self.assertRaises(FileNotFoundError):
            self.s3_utils.download_file(non_existent_key)

    def tearDown(self):
        """
        테스트 후 정리 작업
        """
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir)

        """
        테스트 후 모의 AWS 환경 정리.
        리소스 확보를 위해 모의 AWS 서비스를 중지합니다.
        """
        self.mock.stop()


if __name__ == "__main__":
    unittest.main()
