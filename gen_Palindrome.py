import random

def is_palindrome(number):
    return str(number) == str(number)[::-1]

def generate_palindrome_data(num_examples):
    data = []
    num_palindromes = num_examples // 2
    num_non_palindromes = num_examples - num_palindromes

    # Generate palindrome numbers
    while len(data) < num_palindromes:
        number = random.randint(1, 999999)
        if is_palindrome(number):
            data.append((number, 1))

    # Generate non-palindrome numbers
    while len(data) < num_examples:
        number = random.randint(1, 999999)
        if not is_palindrome(number):
            data.append((number, 0))

    random.shuffle(data)
    return data

def save_data(filename, data):
    with open(filename, 'w') as f:
        for number, label in data:
            f.write(f"{number}\t{label}\n")



if __name__ == "__main__":
    # Generate training data with 200, 2000, and 20000 examples
    for num_examples in [200, 2000, 20000]:
        train_data = generate_palindrome_data(num_examples)
        filename = f"training_data_{num_examples}.txt"
        save_data(filename, train_data)
        print(f"Saved {num_examples} training examples to {filename}")

    test_size = 100
    test_data = generate_palindrome_data(test_size)
    test_file = f"testing_data_{test_size}.txt"
    save_data(test_file, test_data)
    print(f"Saved {test_size} testing examples to {test_file}") 
