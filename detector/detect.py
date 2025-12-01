from detector.detector_model import FakeNewsDetector

def main():
    detector = FakeNewsDetector()
    detector.load()

    text = input("Enter news text: ")
    result = detector.predict(text)
    print("Prediction:", "FAKE" if result == 1 else "REAL")

if __name__ == "__main__":
    main()
