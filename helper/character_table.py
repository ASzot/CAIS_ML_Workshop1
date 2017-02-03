import numpy as np

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.
        # Arguments
        chars:
        Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        X = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)

def gen_numbers(train_size, digits):
    max_len = digits + 1 + digits
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < train_size:
        f = lambda: int(''.join(np.random.choice(list('0123456789'))
            for i in range(np.random.randint(1, digits + 1))))
        a, b = f(), f()

        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)

        q = '{}+{}'.format(a, b)
        query = q + ' ' * (max_len - len(q))
        ans = str(a + b)
        ans += ' ' * (digits + 1 - len(ans))
        # Don't invert the digit.
        #query = query[::-1]
        questions.append(query)
        expected.append(ans)

    return (questions, expected)
