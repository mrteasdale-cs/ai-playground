from textblob.classifiers import NaiveBayesClassifier

train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('What a wonderful experience!', 'pos'),
    ('I am so happy with the service.', 'pos'),
    ('This is the best day ever!', 'pos'),
    ('I really enjoyed the movie.', 'pos'),
    ('The food was delicious.', 'pos'),
    ('I am feeling great today.', 'pos'),
    ('Such a fantastic event!', 'pos'),
    ('I am pleased with the results.', 'pos'),
    ('This product is excellent.', 'pos'),
    ('I am delighted with the outcome.', 'pos'),
    ('Everything is perfect.', 'pos'),
    ('I am thrilled with the performance.', 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ("My boss is horrible.", 'neg'),
    ('This is the worst experience.', 'neg'),
    ('I am very disappointed.', 'neg'),
    ('The service was terrible.', 'neg'),
    ('I hate this place.', 'neg'),
    ('This is unacceptable.', 'neg'),
    ('I am frustrated with the quality.', 'neg'),
    ('The food was awful.', 'neg'),
    ('I am not happy with this.', 'neg'),
    ('This is a bad product.', 'neg'),
    ('I regret buying this.', 'neg'),
    ('The event starts at 7 PM.', 'neu')
]

test = [
    ('I really like this place.', 'pos'),
    ('The service was fantastic.', 'pos'),
    ('I am very pleased with the results.', 'pos'),
    ('This is a great product.', 'pos'),
    ('I am happy with my purchase.', 'pos'),
    ('The movie was enjoyable.', 'pos'),
    ('I love the atmosphere here.', 'pos'),
    ('I do not like the taste.', 'neg'),
    ('I do not like the way I have to do this job.', 'neg'),
    ('The experience was bad.', 'neg'),
    ('I am unhappy with the service.', 'neg'),
    ('This is disappointing.', 'neg'),
    ('I hate waiting in line.', 'neg'),
    ('The food was not good.', 'neg'),
    ('I am frustrated with the delay.', 'neg'),
    ('This is unacceptable behavior.', 'neg'),
    ('The weather is cloudy today.', 'neu'),
    ('I have an appointment later.', 'neu'),
    ('The package was delivered.', 'neu'),
    ('He is working on the project.', 'neu'),
    ('She is reading a book.', 'neu'),
    ('The meeting is scheduled for noon.', 'neu'),
    ('I will call you tomorrow.', 'neu'),
    ('The event was okay.', 'neu'),
    ('This is an average day.', 'neu'),
    ('I am going to the gym.', 'neu'),
    ('The product arrived on time.', 'neu'),
    ('I am neutral about this.', 'neu'),
    ('The service was acceptable.', 'neu'),
    ('I feel indifferent.', 'neu'),
    ('This is neither good nor bad.', 'neu')
]

cl = NaiveBayesClassifier(train)

def classify_batch(sentences):
    print("Batch classification results:")
    for sentence in sentences:
        label = cl.classify(sentence)
        prob_dist = cl.prob_classify(sentence)
        print(f"Text: {sentence}")
        print(f"Predicted: {label}")
        print("Probabilities:")
        for cat in prob_dist.samples():
            print(f"  {cat}: {prob_dist.prob(cat):.2f}")
        print("-" * 30)

def show_informative_features(n=10):
    print(f"\nTop {n} most informative features:")
    cl.show_informative_features(n)

def evaluate_classifier(test_data):
    correct = 0
    total = len(test_data)
    print("\nDetailed test results:")
    for text, actual in test_data:
        predicted = cl.classify(text)
        result = "✓" if predicted == actual else "✗"
        print(f"{result} Text: {text}")
        print(f"   Actual: {actual}, Predicted: {predicted}")
    accuracy = cl.accuracy(test_data)
    print(f"\nOverall accuracy: {accuracy:.2f}")

def interactive_mode():
    print("\nEnter a sentence to classify (type 'exit' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        label = cl.classify(user_input)
        print(f"Predicted sentiment: {label}")

if __name__ == "__main__":
    print("Single classification example:")
    print(cl.classify("I like this place!"))
    print(f"Accuracy: {cl.accuracy(test):.2f}")

    # Batch classification
    batch_examples = [
        "This is the best!",
        "I don't like this at all.",
        "The weather is fine.",
        "Absolutely wonderful!",
        "This is disappointing."
    ]
    classify_batch(batch_examples)

    show_informative_features(5)

    evaluate_classifier(test)

    interactive_mode()
