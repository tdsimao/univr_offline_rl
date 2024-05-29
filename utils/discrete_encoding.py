

class EncoderDecoder:
    def __init__(self, domains: list):
        self.domains = domains
        self.size = self.compute_size()

    def compute_size(self):
        size = 1
        for d in self.domains:
            size *= len(d)
        return size

    def encode(self, *args):
        res = 0
        for value, domain in zip(args, self.domains):
            res *= len(domain)
            res += value
        return res

    def decode(self, i):
        out = []
        for d in reversed(self.domains):
            out.append(i % len(d))
            i = i // len(d)
        # assert 0 <= i < len(self.domains[0])
        return list(reversed(out))
