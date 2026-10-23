import os
import shutil
import tempfile
import unittest
from io import BytesIO


fd, db_path = tempfile.mkstemp(suffix=".db")
os.close(fd)
upload_dir = tempfile.mkdtemp()
os.environ["DATABASE_PATH"] = db_path
os.environ["UPLOAD_FOLDER"] = upload_dir
os.environ["CONTRACT_ADDRESS"] = ""
os.environ["WEB3_PROVIDER"] = ""

from backend.app import app  # noqa: E402


class ApiSecurityTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        response = self.client.post(
            "/api/auth/register",
            json={"name": "Test User", "email": "test@example.com", "password": "password123"},
        )
        if response.status_code == 409:
            response = self.client.post(
                "/api/auth/login",
                json={"email": "test@example.com", "password": "password123"},
            )
        self.headers = {"Authorization": f"Bearer {response.json['token']}"}

    def test_private_dashboard_requires_auth(self):
        self.assertEqual(self.client.get("/api/dashboard").status_code, 401)
        self.assertEqual(self.client.get("/api/dashboard", headers=self.headers).status_code, 200)

    def test_upload_validation_and_chain_history(self):
        bad = self.client.post(
            "/api/upload",
            headers=self.headers,
            data={"file": (BytesIO(b"bad"), "report.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(bad.status_code, 400)

        good = self.client.post(
            "/api/upload",
            headers=self.headers,
            data={"file": (BytesIO(b"pngdata"), "report.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(good.status_code, 200)

        report_id = good.json["report_id"]
        file_hash = good.json["sha256_hash"]
        anchored = self.client.post(
            "/api/chain/storehash",
            headers=self.headers,
            json={"report_id": report_id, "prediction_hash": file_hash},
        )
        self.assertEqual(anchored.status_code, 200)
        self.assertEqual(len(self.client.get("/api/reports", headers=self.headers).json), 1)
        self.assertEqual(len(self.client.get("/api/chain/transactions", headers=self.headers).json), 1)

    @classmethod
    def tearDownClass(cls):
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            pass
        shutil.rmtree(upload_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
