from fastapi.testclient import TestClient
import pprint

from app.main import app

client = TestClient(app)

def test_analyze_basketball_front():
    json = {
        "analysisCategory": "front-basketball",
        "analysisName": "first analysis",
        "analysisSummary": {},
        "owner": "oQMc41YJo7PJFJsj8i2h",
        "sourceVideoPath": "source-videos/basketball_front_test.MOV",
        "outputVideoPath": ""
    }
    response = client.post("/analyze_basketball_front/", json=json)
    pprint.pprint(response.json())
    assert response.status_code == 200