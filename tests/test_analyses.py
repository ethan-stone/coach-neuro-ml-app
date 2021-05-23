from app import analyses, utilities


def test_basketball_front():
    video_path = "source-videos/basketball_front_test.MOV"

    summary = analyses.basketball_front(video_path)    