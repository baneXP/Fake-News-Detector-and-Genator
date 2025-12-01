from detector.detector_model import FakeNewsDetector

if __name__ == "__main__":
    detector = FakeNewsDetector()
    detector.train("data/train.csv")
