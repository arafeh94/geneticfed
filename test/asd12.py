import math


def load_attempts_needed(probability_success, desired_probability):
    n = math.ceil(math.log(1 - desired_probability) / math.log(1 - probability_success))
    return n


probability_success = 1/52

desired_probability = 0.99

attempts_needed = load_attempts_needed(probability_success, desired_probability)

# Print the result
print(
    f"You need to try {attempts_needed} times to achieve a {desired_probability * 100}% probability of working at least once.")
