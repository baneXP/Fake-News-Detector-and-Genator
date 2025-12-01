from generator.generator_model import FakeNewsGenerator

def main():
    gen = FakeNewsGenerator()
    prompt = "Breaking news: Scientists discovered"
    print("Generated Fake News:")
    print(gen.generate(prompt, max_length=80))

if __name__ == "__main__":
    main()
